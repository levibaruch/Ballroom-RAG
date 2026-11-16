"""JudgeClaims: every cited claim of the answer against the blocks it cites, by the decision model.

There is no evidence check BEFORE the answer: it cannot know which 1-3 blocks the writer will use.
The answer is written first; unsupported claims then send the graph to FillGap or a rewrite.
"""
from concurrent.futures import ThreadPoolExecutor

from dance_rag_agentic.config import DECISION_PARALLEL_CLOUD, JUDGE_CLAIMS_PER_CALL, JUDGE_SUPPORTED_FROM, MODELS
from dance_rag_agentic.agent.evidence import EVIDENCE_BRACKET, split_claims, table_units, unit_label, unit_text
from dance_rag_agentic.agent.decision import answers_question_probability, choices
from dance_rag_agentic.agent.prompts import CLAIM_CRITERIA, CLAIM_INSTRUCTIONS


def judge_claims(answer):
    """Each cited claim against the tables it cites. Claims are grouped by the tables they cite and judged in
    small batches: each call gets only its own tables (whole chart sides / whole prose tables, once, cited rows
    marked '>>') and at most JUDGE_CLAIMS_PER_CALL claims. One big call (21 claims over two whole figure charts)
    ran the decision model out of memory.
    Returns [{'claim', 'ids', 'verdict', 'p_supported', 'probabilities', 'judge_input'}]; verdict = SUPPORTED /
    CONTRADICTED / NOT_IN_EVIDENCE (prompts.CLAIM_CRITERIA); claims without ids are skipped (CheckCites does those)."""
    claims = [claim for claim in split_claims(answer) if claim["ids"]]
    groups = {}                                              # {tuple of tables: [claim, ...]}
    for claim in claims:
        tables = tuple(table_units(claim["ids"], scope="section"))
        groups.setdefault(tables, []).append(claim)
    batches = []                                             # (evidence text, claims): one decision call each
    for tables, group in groups.items():
        cited = [evidence_id for claim in group for evidence_id in claim["ids"]]
        evidence_text = "\n\n".join(f"({unit_label(unit)})\n{unit_text(unit, marked)}"
                                    for unit, marked in table_units(cited, scope="section").items())
        batches += [(evidence_text, group[start:start + JUDGE_CLAIMS_PER_CALL])
                    for start in range(0, len(group), JUDGE_CLAIMS_PER_CALL)]

    def judge_batch(batch):
        evidence_text, batch_claims = batch
        questions = {f"claim_{number}": CLAIM_INSTRUCTIONS.format(
                         ids=", ".join(f"[{section_id}.{block}]" for section_id, block in claim["ids"]),
                         claim=claim["claim"])
                     for number, claim in enumerate(batch_claims)}
        answers = choices(f"EVIDENCE:\n{evidence_text}", questions, CLAIM_CRITERIA)
        return [claim | verdict_of(answers[f"claim_{number}"]) | dict(judge_input=evidence_text)
                for number, claim in enumerate(batch_claims)]

    judged = [claim for result in run_parallel(judge_batch, batches) for claim in result]
    return _in_answer_order(judged, claims)


def decision_parallel():
    """How many decision calls may run at once: DECISION_PARALLEL_CLOUD for a cloud model, 1 for a local one."""
    from dance_rag_agentic.agent.models import is_cloud
    return DECISION_PARALLEL_CLOUD if is_cloud(MODELS.decision) else 1


def run_parallel(function, items):
    """[function(item) for item in items], up to decision_parallel() at a time; results in the order of items.
    The calls are independent HTTP requests to the decision model, so the verdicts are the same as one by one."""
    workers = min(decision_parallel(), len(items))
    if workers <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


def _in_answer_order(judged, claims):
    position = {(claim["claim"], tuple(claim["ids"])): index for index, claim in enumerate(claims)}
    return sorted(judged, key=lambda claim: position[(claim["claim"], tuple(claim["ids"]))])


def answers_question(question, answer):
    """P(the answer responds to everything the question asks), judged on the answer text without its ids."""
    return round(answers_question_probability(question, EVIDENCE_BRACKET.sub("", answer).strip()), 2)


def verdict_of(answer):
    """The decision model's choice -> verdict. A SUPPORTED below JUDGE_SUPPORTED_FROM is too unsure to accept:
    it counts as NOT_IN_EVIDENCE (a low-confidence verdict is downgraded, as in calibrated fact checking)."""
    probabilities = {label: round(p, 2) for label, p in answer.get("probabilities", {}).items()}
    p_supported = probabilities.get("SUPPORTED", 0.0)
    verdict = answer["choice"]
    if verdict == "SUPPORTED" and p_supported < JUDGE_SUPPORTED_FROM:
        verdict = "NOT_IN_EVIDENCE"
    return dict(verdict=verdict, p_supported=p_supported, probabilities=probabilities)


def unsupported(judged_claims):
    """Claims that are not SUPPORTED (old results without a verdict: by probability only)."""
    return [claim for claim in judged_claims
            if claim.get("verdict", "SUPPORTED") != "SUPPORTED" or claim["p_supported"] < JUDGE_SUPPORTED_FROM]
