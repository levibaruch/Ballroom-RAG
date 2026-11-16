"""The decision model (clef:cloud by default; any model served through Ollama's /v1/systemone endpoint).

It never writes text. It answers typed questions about a piece of text (the 'state'):
  choice -> one label from a fixed list, with a probability per label and a confidence (0-1)
  noul   -> yes/no, returned as P(yes)
One call can ask up to 64 named questions about the same state.

The label descriptions (prompts.py) ARE the prompt, after three tuning rounds on 42 draft questions.
Their examples are deliberately not taken from those test questions.
"""
import time

import requests

from dance_rag_agentic.retrieval.books import books_sentence, library
from dance_rag_agentic.config import DECISION_MAX_STATE_CHARACTERS, KEEP_ALIVE, MODELS, OLLAMA_URL, CloudAccessError
from dance_rag_agentic.agent.usage import record_decision_usage

from dance_rag_agentic.agent.prompts import (ANSWERS_QUESTION_CRITERIA, ANSWERS_QUESTION_INSTRUCTIONS,
                                             ASPECT_DESCRIPTIONS, ASPECT_INSTRUCTIONS, KIND_DESCRIPTIONS,
                                             KIND_INSTRUCTIONS, MIXED_CRITERIA, MIXED_INSTRUCTIONS)


def kind_instructions():
    return KIND_INSTRUCTIONS.format(count=len(library().dances), books=books_sentence())


def decide(state, questions):
    """One decision-model call. questions: {name: {"type": "choice"|"noul", "instructions": ..., "criteria": ...}}.
    Returns {name: answer}; choice answer = {"choice", "probabilities", "confidence"}, noul answer = {"noul": P(yes)}."""
    request_body = {"model": MODELS.decision, "state": state[:DECISION_MAX_STATE_CHARACTERS], "questions": questions,
                    "keep_alive": KEEP_ALIVE}            # stay loaded while the answer model works
    for attempt in (1, 2):                  # Ollama can drop a connection while it (un)loads a model: retry once
        try:
            response = requests.post(f"{OLLAMA_URL}/v1/systemone", json=request_body, timeout=300)
            response.raise_for_status()
            response_json = response.json()
            record_decision_usage(response_json, MODELS.decision)
            return response_json["answers"]
        except (requests.RequestException, KeyError, ValueError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status in (401, 402, 403):                  # account problem: a retry cannot help
                raise CloudAccessError(f"decision model {MODELS.decision}: HTTP {status} "
                                       f"{getattr(error.response, 'text', '')[:200]}") from error
            if attempt == 2:
                raise DecisionModelError(f"{MODELS.decision}: {type(error).__name__}: {error}"[:300]) from error
            time.sleep(5)


class DecisionModelError(RuntimeError):
    """The decision model did not answer twice. Nodes catch this and carry on without the decision."""


def classify_question(question):
    """Kind + aspect + mixed in one call: {'kind': {...choice answer...}, 'aspect': {...choice answer...},
    'mixed': {'noul': P(technical part AND general-advice part)}}."""
    return decide(question, {
        "kind": {"type": "choice", "instructions": kind_instructions(), "criteria": KIND_DESCRIPTIONS},
        "aspect": {"type": "choice", "instructions": ASPECT_INSTRUCTIONS, "criteria": ASPECT_DESCRIPTIONS},
        "mixed": {"type": "noul", "instructions": MIXED_INSTRUCTIONS, "criteria": MIXED_CRITERIA},
    })


def yes_probabilities(state, yes_no_questions, criteria=None):
    """Several yes/no questions about one state in one call: {name: instructions} -> {name: P(yes)}.
    criteria: {"true": what yes means, "false": what no means}, shared by all questions (optional)."""
    questions = {name: {"type": "noul", "instructions": instructions} | ({"criteria": criteria} if criteria else {})
                 for name, instructions in yes_no_questions.items()}
    answers = decide(state, questions)
    return {name: answer["noul"] for name, answer in answers.items()}


def choices(state, choice_questions, labels):
    """Several choice questions with the same labels about one state in one call:
    {name: instructions} -> {name: {"choice", "probabilities", "confidence"}}."""
    questions = {name: {"type": "choice", "instructions": instructions, "criteria": labels}
                 for name, instructions in choice_questions.items()}
    return decide(state, questions)


def answers_question_probability(question, answer):
    """P(the answer responds to everything the question asks). Short input: question + answer only."""
    return yes_probabilities(f"QUESTION: {question}\n\nANSWER:\n{answer}",
                             {"answers": ANSWERS_QUESTION_INSTRUCTIONS}, ANSWERS_QUESTION_CRITERIA)["answers"]
