"""Collection. Talks to providers, writes the cache, and does nothing else.

Scoring lives in score.py and never runs here. A scoring bug must never cost a
re-collection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from . import __version__
from .adapters.base import Question
from .cache import ResponseCache, build_record, make_cache_key
from .config import RunConfig
from .images import (
    build_manifest,
    load_and_encode,
    manifest_hash,
    manifest_to_json,
    referenced_cameras,
)
from .prompts import IMAGE_CONDITIONS, format_options_block, get_prompt
from .providers.base import RetryableError, TerminalError

# Terminal-error reasons that describe the *setup*, not the request. These are
# identical for every question, so caching them per question would be both
# useless and actively harmful — terminal records are never retried.
CONFIG_ERROR_REASONS = frozenset({"missing_credential", "auth", "unknown_model"})


class ConfigurationError(RuntimeError):
    """Raised when collection cannot proceed for reasons unrelated to any question."""


@dataclass
class CollectionStats:
    considered: int = 0
    cached_hits: int = 0
    calls_made: int = 0
    successes: int = 0
    terminal_errors: int = 0
    retryable_failures: int = 0
    measured_usd: float = 0.0
    budget_exhausted: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def git_sha(default: str = "") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return default


def git_is_dirty(default: bool = True) -> bool:
    """Uncommitted code cannot be reconstructed from a recorded commit SHA."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return default


def render_prompt(question: Question, cfg: RunConfig) -> tuple[str, str, str]:
    template = get_prompt(cfg.prompt_key)
    stem = question.stem_for(cfg.condition)
    options_block = format_options_block(question.letters, question.option_texts)
    system, user = template.render(question=stem, options_block=options_block)
    return system, user, template.hash


def collect(
    questions: list[Question],
    cfg: RunConfig,
    provider,
    cache: ResponseCache,
    *,
    image_root: str = "",
    dry_run: bool = False,
    limit: int | None = None,
    progress_every: int = 25,
    verbose: bool = True,
    cohort_id: str = "",
    max_usd: float | None = None,
    requests_per_minute: float = 0.0,
) -> CollectionStats:
    stats = CollectionStats()
    if max_usd is not None:
        if max_usd <= 0:
            raise ValueError("max_usd must be positive")
        if cfg.model.usd_per_1m_input is None or cfg.model.usd_per_1m_output is None:
            raise ValueError("max_usd requires input and output prices")
    if requests_per_minute < 0:
        raise ValueError("requests_per_minute cannot be negative")
    sha = git_sha()
    needs_images = cfg.condition in IMAGE_CONDITIONS
    min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
    last_call_started: float | None = None

    todo = questions[:limit] if limit else questions

    # Resolve and hash every required image before the first paid call. Doing
    # this lazily inside the request loop can spend hundreds of calls before a
    # missing side/rear view is discovered late in the cohort.
    prepared_images: dict[str, tuple[list, str]] = {}
    if needs_images:
        for q in todo:
            try:
                refs = build_manifest(
                    q.image_paths,
                    root=image_root,
                    cameras=referenced_cameras(q.stem),
                )
            except FileNotFoundError as exc:
                raise ConfigurationError(
                    f"image preflight failed for {q.question_id}: {exc}"
                ) from exc
            prepared_images[q.question_id] = (
                refs,
                manifest_hash(refs, cfg.corruption, cfg.corruption_severity),
            )

    for i, q in enumerate(todo, start=1):
        stats.considered += 1

        system, user, prompt_hash = render_prompt(q, cfg)

        refs = []
        mhash = ""
        if needs_images:
            refs, mhash = prepared_images[q.question_id]

        key = make_cache_key(
            key_fields=cfg.key_fields(),
            question_id=q.question_id,
            prompt_hash=prompt_hash,
            image_manifest_hash=mhash,
        )

        if cache.has(key):
            stats.cached_hits += 1
            continue

        if dry_run:
            if verbose and i == 1:
                print("--- dry run: first rendered prompt ---")
                print(f"[system]\n{system}\n")
                print(f"[user]\n{user}\n")
                print(f"[images] {[r.camera for r in refs]}")
                print(f"[cache_key] {key}")
                print("--- end ---")
            continue

        if max_usd is not None and stats.measured_usd >= max_usd:
            stats.budget_exhausted = True
            if verbose:
                print(
                    f"  stop: measured spend ${stats.measured_usd:.6f} reached "
                    f"the ${max_usd:.6f} ceiling",
                    file=sys.stderr,
                )
            break

        images = None
        if refs:
            images = load_and_encode(
                refs,
                corruption=cfg.corruption,
                severity=cfg.corruption_severity,
                seed=cfg.seed,
            )

        if min_interval and last_call_started is not None:
            remaining = min_interval - (time.monotonic() - last_call_started)
            if remaining > 0:
                time.sleep(remaining)
        last_call_started = time.monotonic()
        stats.calls_made += 1
        try:
            resp = provider.complete(
                system=system,
                user=user,
                images=images,
                model_string=cfg.model.model_string,
                temperature=cfg.decode.temperature,
                top_p=cfg.decode.top_p,
                max_tokens=cfg.decode.max_tokens,
                seed=cfg.seed,
                thinking_effort=cfg.decode.thinking_effort,
            )
        except TerminalError as exc:
            # Configuration errors are not per-question outcomes. A missing or
            # rejected credential says nothing about this question and will
            # affect every remaining one identically, so caching it would
            # poison the cache: terminal records are never retried, and the
            # run could never be repaired without deleting the file by hand.
            # Abort loudly instead, before burning through the whole sample.
            if exc.reason in CONFIG_ERROR_REASONS:
                raise ConfigurationError(
                    f"aborting after {stats.considered} question(s): {exc}. "
                    "This is a local configuration problem, not a per-question "
                    "failure, so nothing was cached and no money was spent. "
                    "Fix it and rerun."
                ) from exc

            # Everything else is genuinely about this request. Cached as
            # terminal, because retrying a malformed request or a refusal on
            # every resume burns budget on a call that can never succeed.
            stats.terminal_errors += 1
            cache.append(
                build_record(
                    cache_key=key, status="terminal_error", question=q, cfg=cfg,
                    prompt_hash=prompt_hash, system=system, user=user,
                    error_reason=exc.reason, error_message=str(exc),
                    image_manifest=manifest_to_json(refs),
                    harness_version=__version__, git_sha=sha, cohort_id=cohort_id,
                )
            )
            continue
        except RetryableError as exc:
            # NOT cached. Next run picks it up again.
            stats.retryable_failures += 1
            if verbose:
                print(f"  retryable failure on {q.question_id}: {exc}", file=sys.stderr)
            continue

        stats.successes += 1
        usage = resp.usage or {}
        if cfg.model.usd_per_1m_input is not None:
            stats.measured_usd += (
                float(usage.get("prompt_tokens") or 0)
                * cfg.model.usd_per_1m_input / 1_000_000
            )
        if cfg.model.usd_per_1m_output is not None:
            stats.measured_usd += (
                float(usage.get("completion_tokens") or 0)
                * cfg.model.usd_per_1m_output / 1_000_000
            )
        cache.append(
            build_record(
                cache_key=key, status="success", question=q, cfg=cfg,
                prompt_hash=prompt_hash, system=system, user=user, response=resp,
                image_manifest=manifest_to_json(refs),
                harness_version=__version__, git_sha=sha, cohort_id=cohort_id,
            )
        )

        if verbose and progress_every and stats.calls_made % progress_every == 0:
            print(
                f"  [{i}/{len(todo)}] calls={stats.calls_made} "
                f"ok={stats.successes} terminal={stats.terminal_errors} "
                f"retryable={stats.retryable_failures}",
                file=sys.stderr,
            )

    stats.measured_usd = round(stats.measured_usd, 8)
    return stats


def append_run_log(path: str, record: dict) -> None:
    """Append one credential-free collection receipt and make it durable."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "schema": 1,
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **record,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
