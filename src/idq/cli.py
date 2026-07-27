"""Command line entry points.

    python -m idq.cli inspect  --data data/drivelm/v1_1_train_nus.json
    python -m idq.cli collect  --adapter fixture --provider mock --condition blind_tags
    python -m idq.cli score    --cache results/cache.jsonl --out results/scored.jsonl
    python -m idq.cli analyze  --scored results/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from .adapters import DriveLMAdapter, FixtureAdapter, summarize
from .analyze import summarize_run
from .cache import ResponseCache
from .config import MOCK_MODEL, DecodeParams, ModelSpec, RunConfig
from .collect import collect
from .providers import MockProvider, OpenAICompatProvider
from .score import read_rows, score_records, write_rows


def _adapter(args):
    if args.adapter == "drivelm":
        if not args.data:
            sys.exit("--data is required for the drivelm adapter")
        return DriveLMAdapter(path=args.data)
    return FixtureAdapter(n=args.n_fixture, seed=args.fixture_seed)


def _provider(args):
    if args.provider == "mock":
        return MockProvider(style=args.mock_style, seed=args.seed, fail_rate=args.mock_fail_rate)
    if not args.base_url:
        sys.exit("--base-url is required for a live provider")
    return OpenAICompatProvider(base_url=args.base_url, api_key_env=args.api_key_env)


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
        served_by=args.provider,
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


def cmd_collect(args) -> None:
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
        ),
        corruption=args.corruption,
        corruption_severity=args.severity,
    )

    cache = ResponseCache(args.cache)
    stats = collect(
        questions, cfg, _provider(args), cache,
        image_root=args.image_root, dry_run=args.dry_run, limit=args.limit,
    )
    print(json.dumps({"config": cfg.key_fields(), "stats": stats.as_dict(),
                      "cache_size": len(cache)}, indent=2, default=str))


def cmd_pilot(args) -> None:
    from .pilot import run_pilot

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
    adapter = _adapter(args)
    questions = adapter.load()
    cache = ResponseCache(args.cache)
    rows = score_records(cache.records(), questions)
    write_rows(rows, args.out)
    print(json.dumps({"scored_rows": len(rows), "out": args.out}, indent=2))


def cmd_analyze(args) -> None:
    rows = read_rows(args.scored)
    print(summarize_run(rows).to_json())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="idq")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--adapter", default="fixture", choices=["fixture", "drivelm"])
        sp.add_argument("--data", default="")
        sp.add_argument("--n-fixture", type=int, default=400)
        sp.add_argument("--fixture-seed", type=int, default=1234)

    sp = sub.add_parser("inspect"); common(sp); sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("collect"); common(sp)
    sp.add_argument("--provider", default="mock", choices=["mock", "openai_compat"])
    sp.add_argument("--base-url", default="")
    sp.add_argument("--api-key-env", default="IDQ_API_KEY")
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
    sp.add_argument("--cache", default="results/cache.jsonl")
    sp.add_argument("--image-root", default="")
    sp.add_argument("--sample", type=int, default=0)
    sp.add_argument("--sample-seed", type=int, default=7)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--mock-style", default="uniform", choices=["uniform", "messy"])
    sp.add_argument("--mock-fail-rate", type=float, default=0.0)
    sp.add_argument("--usd-in", type=float, default=None)
    sp.add_argument("--usd-out", type=float, default=None)
    sp.add_argument("--price-date", default="")
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("pilot"); common(sp)
    sp.add_argument("--provider", default="mock", choices=["mock", "openai_compat"])
    sp.add_argument("--base-url", default="")
    sp.add_argument("--api-key-env", default="BASETEN_API_KEY")
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
    sp.add_argument("--cache", default="results/cache.jsonl")
    sp.add_argument("--out", default="results/scored.jsonl")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("analyze")
    sp.add_argument("--scored", default="results/scored.jsonl")
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
