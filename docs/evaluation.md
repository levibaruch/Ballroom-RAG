# Evaluation

## The gold set

129 questions, each with a reference answer and its sources (book, page, section, a short quote). They were written
**without** using the system: writers read the books directly. Every item is reviewed by hand and marked `gold`
or `rejected`. The set itself is kept private, because its reference answers and quotes come from copyrighted books.
The format, with made-up placeholder values: `dance_rag_agentic/evaluation/gold/example/questions.json`.

Writing rules: one writer per book, 10 questions each with a fixed mix of kinds, plus a set of unrelated questions.
Each writer had to include typos, other languages and phrasing that differs from the book's own terms. Answers come
only from the book text the writer read, with an exact short quote per source so a reviewer can check it quickly.

| Kind | Questions | What it tests |
|---|---|---|
| FIGURE | 43 | one figure: a step detail (timing, footwork, alignment, turn, lead, hips, rise/sway) or the whole figure |
| FIGURE_COMPARE | 17 | man vs lady, two figures, or the same figure in two dances |
| TECHNIQUE | 26 | principles, music, character, leading and following, list questions |
| TERM | 18 | what an abbreviation, term, position or notation means |
| CHARACTER | 7 | the feel of a dance |
| ABOUT | 4 | questions about the assistant itself |
| GENERAL | 2 | learner advice (answered from general knowledge, never as book content) |
| OUT_OF_SCOPE | 12 | other dance styles, history, rules, non-dance topics |

Variations across the set: short (38), typos (30), Dutch (17), prompt injections (15), all lowercase (13),
traps with a wrong premise or a figure that is not in the book (13), long or chatty phrasing (20).

## Scores (code-only, `make eval`)

| Score | Meaning |
|---|---|
| `route_ok` | the route matches the expected kind (UNSURE is counted separately; a trap is scored on the kind it looks like) |
| `section_hit` | at least one cited section is one of the gold sources' sections |
| `page_hit` | at least one cited (dance, page) is a gold source page |
| `refusal_ok` | out-of-scope questions: no citations, and a refusal or "not found" |
| `general_ok` | general-advice questions: tips and no citations |
| unsupported claims | claims the judge did not accept that are still in the final answer |
| tokens, cost | per model, per question, from the servers' usage reports |

Each question also writes a full trace (`cache/agentic/traces/`), so a failing score can be read back step by step.

Quality questions (is this answer correct, complete and useful to a learner?) are settled by a **blind review**:
the answers from two variants are shuffled per question and scored 1–5 on correctness, completeness and learner
quality. Prompt changes, search output formats and index designs were all decided this way (`decisions.md`).

## Results

| Run | Questions | Result |
|---|---|---|
| Index v4 vs v1 / v3 (2026-11-13) | 29 | gold page cited 27/27, 0 unsupported claims, blind review 11.45/15 (v1 11.28, v3 11.10), $0.25 in total |
| Routing on the drafts (2026-11-13) | 100 | 84 routed right, 12 UNSURE, 4 wrong (with the top-probability < 0.5 rule) |
| Prompt A/B: long vs compact (2026-11-14) | 10 | long prompts 111 vs 88 points (blind review); compact cost 17% more |
| Baseline before the code cleanup (2026-11-16) | 10, one per kind | route 10/10, gold page 6/6, gold section 4/6, refusal 1/1, general 1/1, ~$0.06 |

Caveat: the decision model's label descriptions were tuned on the draft questions, so routing scores on that set are
optimistic. One run per question is noisy, because the same question can give a better or worse answer on another run.

## Running it

```bash
make eval                                                       # all non-rejected questions, models from settings.toml
uv run python -m dance_rag_agentic.evaluation.evaluate --routing-only      # routing only: ~1 s per question
uv run python -m dance_rag_agentic.evaluation.evaluate --ids ids.txt --shard 0/2 --tag baseline   # a subset, in parallel
uv run python -m dance_rag_agentic.evaluation.gold.validate                # sources exist, quotes appear in their blocks
```

Results go to `cache/agentic/eval/<time>-<models>-index_v4[-tag][-shard].json`. Cloud models cost about 1–2 cents
per question.
