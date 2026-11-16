"""Answering a question: the graph (graph.py) and everything it calls.

    entities.py   ExtractEntities: dances, figures, role, steps (code, fuzzy match against the books' lists)
    decision.py   decision-model client: kind / aspect choices, claim verdicts, 'answers the question?'
    tools.py      the toolbox: graph nodes and agents use the same functions
    agents.py     searching agents and the writers (answer, repair, polish, general advice, ...)
    evidence.py   EvidenceLog, evidence ids, CheckCites, claim splitting, rendering with sources
    checks.py     JudgeClaims: each claim against the blocks it cites, the answer against the question
    prompts.py    every prompt text
    safety.py     fixed reply for questions about self-harm
    models.py     which installed models fit which role; usage.py  tokens and cost per model call
"""
