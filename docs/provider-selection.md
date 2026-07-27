# Provider selection — step 2

**Retrieved 2026-07-24. Everything below has a shelf life; re-verify before collection.**

## Recommendation: Baseten

It is the only one of the three that is (a) actually serving Inkling per-token
right now and (b) documented to return the field RQ2 depends on.

```
base_url      https://inference.baseten.co/v1
model_string  thinkingmachines/inkling
api_key_env   BASETEN_API_KEY      (env var only — never commit it)
```

## The comparison

| | Together | Fireworks | Baseten |
|---|---|---|---|
| Model string | `thinkingmachines/Inkling` | `accounts/fireworks/models/inkling` | `thinkingmachines/inkling` |
| Serving | **"Coming soon to Serverless"** | **On-demand only, serverless not supported** | Model API (per-token) |
| Per-token billing | not yet | no — rent GPUs by the hour | yes |
| `reasoning_tokens` in usage | unverified | not shown in docs | **documented, value 182 in their example** |
| `reasoning_content` field | unverified | yes | yes |
| `reasoning_effort` | yes (claimed) | yes, `none…xhigh` + `thinking` budget | yes, `none…xhigh` |
| Stated context | 256K | 1040K | 256K |
| Hardware | — | — | B200 |

Two things to notice.

**Together announced day-0 support but the model page says "coming soon to
Together's Serverless API."** The blog post and the product page disagree.
Don't build on the blog post.

**Fireworks lists Inkling as "Serverless: Not supported" — on-demand
deployment only.** That means renting dedicated B200-class GPUs by the hour for
a 975B model. For 4,500 short calls that is the wrong billing model by a wide
margin; you would pay for idle GPU time between requests. Fireworks becomes
sensible only if you are running tens of thousands of calls back to back.

**The three providers disagree about the context window** (256K vs 1040K).
Doesn't affect this study — driving MCQs are tiny — but it is a good early
signal that provider metadata for a nine-day-old model is not yet reliable.

## RQ2 is safe

Baseten's published example response for Inkling contains exactly this:

```json
"usage": {
  "completion_tokens": 295,
  "prompt_tokens": 9,
  "completion_tokens_details": { "reasoning_tokens": 182 }
}
```

That is `usage.completion_tokens_details.reasoning_tokens` in the
OpenAI-compatible shape — the field you said RQ2 depends on. No proxy metric is
needed, so the fallback decision you wanted made before collection is: **not
needed, but still implemented.** The harness records three independent signals
so a provider change can't strand you:

1. `reasoning_tokens` from usage — the real measurement
2. `reasoning_chars` — length of the `reasoning_content` field
3. `thinking_chars` — length of any inline `<think>` block

`token_profile()` reports `is_proxy: true` if it ever has to fall back, so the
paper always states which quantity the frontier plot is actually on.

## Quantization is disclosed, and it matters

Baseten's response reports `"model": "inferact/inkling-nvfp4"`. You are being
served **NVFP4**, not BF16. This is good news twice over: it is the only reason
a 975B model is affordable per-token, and it is disclosed rather than guessed.

But it belongs in the paper, stated plainly. NVFP4 is a 4-bit floating point
format; quantization can cost accuracy, and a reviewer comparing your Inkling
numbers to Thinking Machines' published benchmarks will want to know why they
differ. The harness now records `served_model` on every single record, so the
claim is backed by the data rather than by this memo.

## A code change this forced

Both Baseten and Fireworks return the chain of thought in
`message.reasoning_content` **alongside** a populated `content` field. My
original code only read `reasoning_content` when `content` was empty.

Had that shipped, the reasoning text would have been concatenated into the
answer string on some responses — and the answer parser would then have seen
every letter the model considered mid-deliberation. "Let me reconsider, A is
also plausible, B is out" contains three option letters. The parser would have
flagged ambiguity or picked wrong, and the invalid-output rate would have been
a measurement of my bug rather than of the model.

Fixed, and pinned by `tests/test_provider_payloads.py`, which asserts against
Baseten's exact published payload.

## One thing to avoid

Inkling is also hosted on **build.nvidia.com**. Don't use it.

It is a free public service and using it would probably be fine. But you are a
contractor on an NVIDIA program, and your IP boundary says no employer compute.
Running your personal research through your client's inference platform makes
that line ambiguous for no benefit, and "probably fine" is not the standard you
set for yourself in the brief. Baseten costs a few dollars and keeps the
boundary crisp.

## Before you spend anything

1. Log into Baseten and confirm the **actual per-token price** for Inkling.
   I could not find it published. The $1.87 in / $4.68 out figures circulating
   are Thinking Machines' own Tinker platform **at a limited-time 50%
   discount** — a different vendor and a promotional rate. Do not put those in
   the paper as Baseten's price.
2. Record the price and the date in `ModelSpec`. It travels into every cache
   record from there.
3. Run the 20-call pilot (`idq pilot`) before anything larger.

## Sources

- [Baseten — Inkling model library](https://www.baseten.co/library/inkling/) (payload shape, reasoning_tokens, NVFP4)
- [Fireworks — Inkling model page](https://fireworks.ai/models/fireworks/inkling) (serverless not supported)
- [Fireworks — Reasoning docs](https://docs.fireworks.ai/guides/reasoning) (reasoning_content, reasoning_effort)
- [Together — Inkling model page](https://www.together.ai/models/inkling) ("coming soon to Serverless")
- [Thinking Machines — Inkling announcement](https://thinkingmachines.ai/news/introducing-inkling/)
