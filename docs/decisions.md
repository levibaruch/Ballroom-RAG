# Decision log

What was tried, what won, and why. Experiments that lost have been removed from the code. Their results are kept
here, so the reasoning survives without the dead code paths.

Measurement rule: a change to prompts, search output or the index becomes the default only after a comparison on
the same gold questions, with a blind review where quality is the question.

## Architecture

| Decision | Instead of | Why |
|---|---|---|
| **Constrained graph**: code holds the control flow (a plain-Python state machine). LLM agents only search, with bounded tool budgets | One free agent with all tools | The free agent had too much freedom: unpredictable cost, skipped reading, answered from memory |
| **Decision model** for the multiple-choice and yes/no calls (kind of question, aspect, is this claim supported, does the answer respond to the question). It returns calibrated probabilities | An LLM router / LLM-as-judge with free-text output | Probabilities give thresholds that can be measured (`UNSURE` below p 0.5), and it's fast and cheap per call |
| **Entities in code** (fuzzy match against the books' own figure lists) | Asking the decision model "does this name a figure?" | The model missed 7 of 22 figures. Code found 41 of 42 |
| **Code override rules** on the predicted kind (no figure named → no figure kinds; two figures/dances/roles → comparison) | Trusting the classifier alone | Routing on the draft set went from 81% to 93% with no wrong routes above the threshold |
| **Verify after answering**: each claim is judged against the blocks it cites (supported / contradicted / not in evidence), then targeted repair | An "enough evidence?" check before answering | A pre-answer check can't know which blocks the writer will use. Judging the finished answer can |
| **Reframed claim judge**: "if the evidence is correct, is this claim also correct?" | "Is this claim supported?" plus exception rules | Literal "supported" rejected valid inferences (a 4/4 source supports "not 3/4"). Patching with exceptions made it worse |
| **One block per chart side** (cite the side, name the step in text) | One evidence id per chart step | Step-level ids caused ranges, off-by-one step errors and rewrite loops |
| **Abbreviations spelled out** in the text given to the models | Abbreviations kept as printed | The models misread them. The books' own abbreviation tables drive the expansion |
| **Answers always teach in three parts** (what it is / how to do it / what to watch) | Free-form answers | Consistent, readable answers for learners |
| **General advice is its own route** (practising, nerves, clothing), from the model's own knowledge, never mixed with book content and always labelled | Refusing everything not in the books | Learners ask these questions too. Labelling keeps the book answers trustworthy |
| **One toolbox, two callers**: the same function is a fixed graph node on one route and an agent tool on another | Separate code for nodes and agents | Less code, and the same behaviour in both places |

## Retrieval and index

| Decision | Measured result |
|---|---|
| **Index v4** (built from the parser combination, heading units, stable content ids) replaces v1–v3 | 29 gold questions: gold page cited 27/27 (v1 23/27), unsupported claims 0 (v1 1), blind review 11.45/15 (v1 11.28, v3 11.10), cost $0.25 vs $0.28. See `parser-comparison.md` |
| v2: heading units + Part 1 indexed once per style. v3: + LLM-written summaries/questions per unit as search-only pointers | Offline retrieval (gold page in top 8): v1 0.82, v2 0.84, v3 0.89. End to end, v3's gain was within noise, and it needs an LLM pass over every unit. Dropped for v4 |
| **Hybrid `search` tool**: BM25 + embeddings fused with reciprocal rank fusion (k = 60), book-index hits first, optional prose/chart filter | vs three separate tools (keyword / semantic / index lookup): same calls and cost, one more key section reached, three tools fewer for the agent |
| **Search output "mid"**: one line per figure, strong text hits with the matching sentence, weaker matches listed apart, where a text lives and its size | vs "list" (8 hit lines with snippets): tool calls 60 → 45, time 174 → 158 s, blind quality 74 vs 72. "read" (best hits read at once): +70% cost, agents kept searching. "compact": quality 63 vs 70 |
| Body-text BM25 for chart rows | Rejected: MRR 0.96 → 0.60 |
| Hard metadata filter (dance/figure) on retrieval | Rejected: breaks list and comparison questions. A soft boost through entities instead |
| 70/30 dense/sparse weighting, chunk-size tuning | No gain on this corpus |

## Prompts

| Decision | Measured result |
|---|---|
| **Long prompts with explicit rule lists** stay the default | A ~55% shorter rewrite lost a blind A/B test on 10 gold questions: **88 vs 111** points, better in only 2 of 10, and 17% *more* cost. Without the "never fill a part with your own ideas" rules, the answer model padded thin evidence with invented advice, which triggered more repair rounds |
| Answer in the question's language; leave an empty part out instead of writing "the books do not give…" | Came out of the blind review of the v4 run |

## Models and serving

| Decision | Why |
|---|---|
| Default models in the cloud (answer `gpt-oss:120b` via Ollama Cloud or Cerebras, decision `clef`). A local pair is still supported (`gpt-oss:20b` + a local decision model) | 120b answered better than 20b, at 8–35 s instead of 1–4 min per question, for about 1 cent per question |
| Production parsing is fully local (bibr with MLX OCR, Docling, PyMuPDF) | The books are copyrighted. Book pages never go to a cloud model during the build |
| **Public API shows a whitelist** of what the system did (nodes, timings, probabilities, tool names, claim verdicts), never book text, prompts or tool results | The demo shows how the system works without leaking the books or prompt internals |
| Daily budget cap + per-visitor rate limit + CORS for the demo site only | The public demo costs money per question |
| The index is mounted into the container, never baked into the image or committed | It contains book text |
