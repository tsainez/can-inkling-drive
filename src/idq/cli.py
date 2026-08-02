"""Command line entry points.

    python -m idq.cli inspect  --data data/drivelm/v1_1_train_nus.json
    python -m idq.cli collect  --adapter fixture --provider mock --condition blind_tags
    python -m idq.cli score    --cache results/cache.jsonl --out results/scored.jsonl
    python -m idq.cli analyze  --scored results/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .adapters import DriveLMAdapter, FixtureAdapter, summarize
from .analyze import summarize_run
from .cache import ResponseCache
from .config import MOCK_MODEL, DecodeParams, ModelSpec, RunConfig
from .collect import append_run_log, collect, git_is_dirty, git_sha
from .providers import MockProvider, OpenAICompatProvider
from .score import read_rows, score_records, write_rows


def _adapter(args):
    if args.adapter in ("drivelm", "drivelm_converted"):
        if not args.data:
            sys.exit(f"--data is required for the {args.adapter} adapter")
    if args.adapter == "drivelm":
        return DriveLMAdapter(path=args.data)
    if args.adapter == "drivelm_converted":
        from .convert import DriveLMConvertedAdapter

        return DriveLMConvertedAdapter(
            path=args.data,
            seed=getattr(args, "convert_seed", 20260731),
            n_per_template=getattr(args, "n_per_template", None) or None,
        )
    return FixtureAdapter(n=args.n_fixture, seed=args.fixture_seed)


def _provider(args):
    if args.provider == "mock":
        return MockProvider(style=args.mock_style, seed=args.seed, fail_rate=args.mock_fail_rate)
    if not args.base_url:
        sys.exit("--base-url is required for a live provider")
    extra_body = {}
    if getattr(args, "provider_only", ""):
        route = {
            "only": [args.provider_only],
            "allow_fallbacks": bool(args.allow_provider_fallbacks),
            "require_parameters": True,
        }
        if args.provider_quantization:
            route["quantizations"] = [args.provider_quantization]
        extra_body["provider"] = route
    return OpenAICompatProvider(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        extra_body=extra_body or None,
        reasoning_format=args.reasoning_format,
        include_sampling_params=not args.omit_sampling_params,
        max_tokens_field=args.max_tokens_field,
    )


def _served_by(args) -> str:
    if getattr(args, "served_by", ""):
        return args.served_by
    base_url = (getattr(args, "base_url", "") or "").lower()
    if "baseten.co" in base_url:
        return "baseten"
    if "openrouter.ai" in base_url:
        route = args.provider_only or "unlocked"
        fallback = "fallbacks" if args.allow_provider_fallbacks else "pinned"
        return f"openrouter:{route}:{fallback}"
    return args.provider


def _model(args) -> ModelSpec:
    if args.provider == "mock":
        # Honour explicit prices even on the mock so the sizing math can be
        # rehearsed offline: "if Baseten charges X, how many questions do I get
        # for my budget?" is worth answering before spending anything.
        if getattr(args, "usd_in", None) is not None and getattr(args, "usd_out", None) is not None:
            return ModelSpec(
                label="mock-priced", model_string="mock/uniform", served_by="mock",
                quantization="n/a",
                usd_per_1m_input=args.usd_in, usd_per_1m_output=args.usd_out,
                price_quoted_on=getattr(args, "price_date", ""),
                price_note="hypothetical price applied to mock responses",
            )
        return MOCK_MODEL
    return ModelSpec(
        label=args.model_label or args.model_string,
        model_string=args.model_string,
        served_by=_served_by(args),
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        quantization=args.quantization,
        usd_per_1m_input=args.usd_in,
        usd_per_1m_output=args.usd_out,
        price_quoted_on=args.price_date,
    )


def cmd_inspect(args) -> None:
    adapter = _adapter(args)
    questions = adapter.load()
    report = {"adapter": adapter.name, "summary": summarize(questions)}
    if hasattr(adapter, "stats"):
        report["adapter_stats"] = adapter.stats
    print(json.dumps(report, indent=2))


def _cohort_questions(args):
    """Load exactly the frozen cohort, refusing any selection drift."""
    from .cohort import read_manifest, select_questions

    manifest = read_manifest(args.cohort)
    if args.adapter != manifest["adapter"]:
        sys.exit(
            f"--adapter {args.adapter} does not match cohort adapter {manifest['adapter']}"
        )
    selection = manifest["selection"]
    if args.convert_seed != selection["convert_seed"]:
        sys.exit("--convert-seed does not match the frozen cohort")
    if args.n_per_template not in (0, selection["n_per_template"]):
        sys.exit("--n-per-template does not match the frozen cohort")

    # Reconstruct only the selected balanced pool instead of loading all 28k.
    args.n_per_template = selection["n_per_template"]
    questions = _adapter(args).load()
    return select_questions(questions, manifest, data_path=args.data), manifest


def cmd_cohort(args) -> None:
    from .cohort import build_manifest, write_manifest

    if args.adapter != "drivelm_converted":
        sys.exit("publication cohorts require --adapter drivelm_converted")
    if args.n_per_template <= 0:
        sys.exit("--n-per-template must be positive when freezing a cohort")
    adapter = _adapter(args)
    questions = adapter.load()
    manifest = build_manifest(
        questions,
        data_path=args.data,
        adapter=adapter.name,
        convert_seed=args.convert_seed,
        n_per_template=args.n_per_template,
        created_on=args.created_on,
        repeat_per_joint_cell=args.repeat_per_joint_cell,
    )
    write_manifest(manifest, args.out)
    print(json.dumps({
        "cohort_id": manifest["cohort_id"],
        "out": args.out,
        "audit": manifest["audit"],
        "repeat_subset_n": manifest["repeat_subset"]["n_questions"],
    }, indent=2))


def cmd_collect(args) -> None:
    if args.provider != "mock" and not args.dry_run and git_is_dirty():
        sys.exit(
            "live collection refused: commit the complete study state first so "
            "git_sha identifies the code and cohort exactly"
        )
    manifest = None
    if args.cohort:
        if args.sample:
            sys.exit("--sample cannot be combined with a frozen --cohort")
        questions, manifest = _cohort_questions(args)
    else:
        adapter = _adapter(args)
        questions = adapter.load()
    if args.sample:
        import random
        random.Random(args.sample_seed).shuffle(questions)
        questions = questions[: args.sample]

    cfg = RunConfig(
        model=_model(args),
        condition=args.condition,
        seed=args.seed,
        decode=DecodeParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking_effort=args.thinking_effort,
            reasoning_format=args.reasoning_format,
            include_sampling_params=not args.omit_sampling_params,
            max_tokens_field=args.max_tokens_field,
        ),
        corruption=args.corruption,
        corruption_severity=args.severity,
    )

    cache = ResponseCache(args.cache)
    stats = collect(
        questions, cfg, _provider(args), cache,
        image_root=args.image_root, dry_run=args.dry_run, limit=args.limit,
        cohort_id=(manifest or {}).get("cohort_id", ""),
        max_usd=args.max_usd,
        requests_per_minute=args.requests_per_minute,
    )
    receipt = {
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "cohort_id": (manifest or {}).get("cohort_id", ""),
        "cohort_path": args.cohort,
        "model_label": cfg.model.label,
        "config": cfg.key_fields(),
        "pricing": {
            "usd_per_1m_input": cfg.model.usd_per_1m_input,
            "usd_per_1m_output": cfg.model.usd_per_1m_output,
            "price_quoted_on": cfg.model.price_quoted_on,
        },
        "cache": args.cache,
        "dry_run": args.dry_run,
        "max_usd": args.max_usd,
        "requests_per_minute": args.requests_per_minute,
        "stats": stats.as_dict(),
        "cache_size": len(cache),
    }
    if args.run_log:
        append_run_log(args.run_log, receipt)
    print(json.dumps(receipt, indent=2, default=str))


def cmd_probe(args) -> None:
    from .probe import probe

    if not args.data:
        sys.exit("--data is required")
    print(json.dumps(probe(args.data, n_examples=args.n_examples), indent=2))


def cmd_pilot(args) -> None:
    from .pilot import run_pilot

    if args.provider != "mock" and git_is_dirty():
        sys.exit(
            "live pilot refused: commit the complete study state first so git_sha "
            "identifies the code exactly"
        )

    adapter = _adapter(args)
    questions = adapter.load()
    import random
    random.Random(args.sample_seed).shuffle(questions)

    cfg = RunConfig(
        model=_model(args),
        condition=args.condition,
        seed=args.seed,
        decode=DecodeParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking_effort=args.thinking_effort,
            reasoning_format=args.reasoning_format,
            include_sampling_params=not args.omit_sampling_params,
            max_tokens_field=args.max_tokens_field,
        ),
        corruption=args.corruption,
        corruption_severity=args.severity,
    )
    report = run_pilot(
        questions, cfg, _provider(args), args.cache,
        n=args.n, n_repeat=args.n_repeat, image_root=args.image_root,
        budget_usd=args.budget, n_models=args.n_models, n_conditions=args.n_conditions,
    )
    print(json.dumps(report.as_dict(), indent=2, default=str))


def cmd_score(args) -> None:
    if args.cohort:
        questions, manifest = _cohort_questions(args)
    else:
        adapter = _adapter(args)
        questions = adapter.load()
        manifest = None
    cache = ResponseCache(args.cache)
    allowed = {q.question_id for q in questions}
    records = (r for r in cache.records() if r.get("question_id") in allowed)
    rows = score_records(records, questions)
    write_rows(rows, args.out)
    print(json.dumps({
        "scored_rows": len(rows),
        "cohort_id": (manifest or {}).get("cohort_id", ""),
        "out": args.out,
    }, indent=2))


def _primary(args) -> tuple[str, str] | None:
    """--primary "inkling,qwen3-235b" designates the preregistered comparison.

    It is exempt from Holm correction, so naming it on the command line after
    seeing the results would be p-hacking. It belongs in the runbook before
    collection starts.
    """
    if not args.primary:
        return None
    parts = [p.strip() for p in args.primary.split(",") if p.strip()]
    if len(parts) != 2:
        sys.exit('--primary takes exactly two model labels, e.g. --primary "inkling,other"')
    return (parts[0], parts[1])


def cmd_analyze(args) -> None:
    rows = read_rows(args.scored)
    report = summarize_run(
        rows,
        primary_comparison=_primary(args),
        baseline_condition=args.baseline,
        degraded_condition=args.degraded,
        alpha=args.alpha,
        n_boot=args.n_boot,
    )
    if args.markdown:
        from .tables import render_markdown

        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(report.sections, title=args.title))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(report.to_json())

    print(report.to_json())
    # Warnings go to stderr as well as into the report, so they survive a
    # redirect of stdout into a file nobody reads again.
    for w in report.sections.get("warnings") or []:
        print(f"warning: {w}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="idq")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--adapter", default="fixture",
                        choices=["fixture", "drivelm", "drivelm_converted"])
        sp.add_argument("--data", default="")
        sp.add_argument("--n-fixture", type=int, default=400)
        sp.add_argument("--fixture-seed", type=int, default=1234)
        sp.add_argument("--convert-seed", type=int, default=20260731)
        sp.add_argument("--n-per-template", type=int, default=0)

    sp = sub.add_parser("inspect"); common(sp); sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("cohort"); common(sp)
    sp.add_argument("--created-on", default="2026-07-31")
    sp.add_argument("--repeat-per-joint-cell", type=int, default=5)
    sp.add_argument("--out", default="study/cohorts/drivelm-balanced-600.json")
    sp.set_defaults(func=cmd_cohort)

    sp = sub.add_parser("collect"); common(sp)
    sp.add_argument("--provider", default="mock", choices=["mock", "openai_compat"])
    sp.add_argument("--base-url", default="")
    sp.add_argument("--api-key-env", default="IDQ_API_KEY")
    sp.add_argument("--served-by", default="")
    sp.add_argument("--provider-only", default="")
    sp.add_argument("--provider-quantization", default="")
    sp.add_argument("--allow-provider-fallbacks", action="store_true")
    sp.add_argument("--model-string", default="mock/uniform")
    sp.add_argument("--model-label", default="")
    sp.add_argument("--quantization", default="unknown")
    sp.add_argument("--condition", default="blind_tags",
                    choices=["clean", "blind_tags", "blind_notags", "corrupt"])
    sp.add_argument("--corruption", default="")
    sp.add_argument("--severity", type=int, default=0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--max-tokens", type=int, default=2048)
    sp.add_argument("--thinking-effort", default=None)
    sp.add_argument("--reasoning-format", default="reasoning_effort",
                    choices=["reasoning_effort", "reasoning"])
    sp.add_argument("--omit-sampling-params", action="store_true")
    sp.add_argument("--max-tokens-field", default="max_tokens",
                    choices=["max_tokens", "max_completion_tokens"])
    sp.add_argument("--cache", default="results/cache.jsonl")
    sp.add_argument("--image-root", default="")
    sp.add_argument("--sample", type=int, default=0)
    sp.add_argument("--sample-seed", type=int, default=7)
    sp.add_argument("--cohort", default="")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--max-usd", type=float, default=None,
                    help="stop after measured successful-response cost reaches this ceiling")
    sp.add_argument("--requests-per-minute", type=float, default=0.0,
                    help="pace call starts; 0 disables client-side pacing")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--mock-style", default="uniform", choices=["uniform", "messy"])
    sp.add_argument("--mock-fail-rate", type=float, default=0.0)
    sp.add_argument("--usd-in", type=float, default=None)
    sp.add_argument("--usd-out", type=float, default=None)
    sp.add_argument("--price-date", default="")
    sp.add_argument("--run-log", default="results/run-log.jsonl")
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("probe")
    sp.add_argument("--data", default="")
    sp.add_argument("--n-examples", type=int, default=3)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("pilot"); common(sp)
    sp.add_argument("--provider", default="mock", choices=["mock", "openai_compat"])
    sp.add_argument("--base-url", default="")
    sp.add_argument("--api-key-env", default="BASETEN_API_KEY")
    sp.add_argument("--served-by", default="")
    sp.add_argument("--provider-only", default="")
    sp.add_argument("--provider-quantization", default="")
    sp.add_argument("--allow-provider-fallbacks", action="store_true")
    sp.add_argument("--model-string", default="mock/uniform")
    sp.add_argument("--model-label", default="")
    sp.add_argument("--quantization", default="unknown")
    sp.add_argument("--condition", default="blind_tags",
                    choices=["clean", "blind_tags", "blind_notags", "corrupt"])
    sp.add_argument("--corruption", default="")
    sp.add_argument("--severity", type=int, default=0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--max-tokens", type=int, default=2048)
    sp.add_argument("--thinking-effort", default=None)
    sp.add_argument("--reasoning-format", default="reasoning_effort",
                    choices=["reasoning_effort", "reasoning"])
    sp.add_argument("--omit-sampling-params", action="store_true")
    sp.add_argument("--max-tokens-field", default="max_tokens",
                    choices=["max_tokens", "max_completion_tokens"])
    sp.add_argument("--cache", default="results/pilot.jsonl")
    sp.add_argument("--image-root", default="")
    sp.add_argument("--sample-seed", type=int, default=7)
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--n-repeat", type=int, default=5)
    sp.add_argument("--budget", type=float, default=15.0)
    sp.add_argument("--n-models", type=int, default=5)
    sp.add_argument("--n-conditions", type=int, default=3)
    sp.add_argument("--mock-style", default="uniform", choices=["uniform", "messy"])
    sp.add_argument("--mock-fail-rate", type=float, default=0.0)
    sp.add_argument("--usd-in", type=float, default=None)
    sp.add_argument("--usd-out", type=float, default=None)
    sp.add_argument("--price-date", default="")
    sp.set_defaults(func=cmd_pilot)

    sp = sub.add_parser("score"); common(sp)
    sp.add_argument("--cohort", default="")
    sp.add_argument("--cache", default="results/cache.jsonl")
    sp.add_argument("--out", default="results/scored.jsonl")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("analyze")
    sp.add_argument("--scored", default="results/scored.jsonl")
    sp.add_argument("--primary", default="",
                    help='preregistered primary comparison, e.g. "inkling,other"; '
                         "exempt from Holm correction")
    sp.add_argument("--baseline", default="clean",
                    help="baseline condition for the robustness comparison")
    sp.add_argument("--degraded", default="corrupt",
                    help="degraded condition for the robustness comparison")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--n-boot", type=int, default=10000)
    sp.add_argument("--markdown", default="",
                    help="also write paper-ready Markdown tables to this path")
    sp.add_argument("--json-out", default="", help="also write the JSON report to this path")
    sp.add_argument("--title", default="Results")
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
