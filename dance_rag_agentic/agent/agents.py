"""The LLM parts: four searching agents (same toolbox, different starting prompt), the answer writer (from the
books, with evidence ids), the general-advice writer (own knowledge, no ids) and the question splitter.

Searching agents only COLLECT evidence: their reading tools put sections into the EvidenceLog, and
their text output is a short note for the graph (what was found, what is missing). The answer writer
has no tools: it writes from the evidence the graph hands it and cites evidence ids.
"""
import os
from functools import cache

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent, UsageLimits                         # noqa: E402
from pydantic_ai.exceptions import ModelAuthenticationError, ModelQuotaExceededError, UsageLimitExceeded  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings  # noqa: E402
from pydantic_ai.providers.ollama import OllamaProvider             # noqa: E402
from pydantic_ai.usage import RunUsage                               # noqa: E402

from dance_rag_agentic.retrieval.books import books_sentence  # noqa: E402
from dance_rag_agentic.config import MODELS, OLLAMA_URL, CloudAccessError, api_provider  # noqa: E402
from dance_rag_agentic.agent.tools import AgentDeps, toolbox  # noqa: E402
from dance_rag_agentic.agent.usage import record_answer_usage  # noqa: E402

from dance_rag_agentic.agent.prompts import (ABOUT_PROMPT, ANSWER_PROMPT, CHARACTER_GENERAL_PROMPT,  # noqa: E402
                                             GENERAL_PROMPT, POLISH_PROMPT, REPAIR_PROMPT, SPLIT_PROMPT)

@cache
def prompt(template):
    """Fill in the books from the knowledge base."""
    return template.format(books=books_sentence())


def settings():
    """Model settings for every answer-model call: the gpt-oss reasoning effort (config.REASONING, read at call time)."""
    from dance_rag_agentic import config
    return OpenAIChatModelSettings(openai_reasoning_effort=config.REASONING)


_models = {}                                    # one client per model name, so a switch takes effect at once


def model():
    if MODELS.answer not in _models:
        provider_name, model_name = api_provider(MODELS.answer)
        if provider_name == "cerebras":                 # CEREBRAS_API_KEY from the environment / .env
            from pydantic_ai.providers.cerebras import CerebrasProvider
            _models[MODELS.answer] = OpenAIChatModel(model_name, provider=CerebrasProvider())
        else:
            _models[MODELS.answer] = OpenAIChatModel(MODELS.answer, provider=OllamaProvider(base_url=f"{OLLAMA_URL}/v1"))
    return _models[MODELS.answer]


def searching_agent(prompt):
    # one search tool: the rules' tool names (keyword/semantic search) map onto it
    prompt += ("\n\nYOUR SEARCH TOOL is `search`: it searches exact words, meaning and the book index together "
               "(kind='prose' for explanations).")
    return Agent(model(), deps_type=AgentDeps, tools=toolbox(), instructions=prompt, model_settings=settings(), retries=2)


async def run_counted(agent, request, **run_options):
    """agent.run(...) that adds its tokens to the meter, also when the run fails or hits a limit:
    pydantic-ai fills the RunUsage we pass in while it works."""
    run_usage = RunUsage()
    try:
        return await agent.run(request, usage=run_usage, **run_options)
    except (ModelAuthenticationError, ModelQuotaExceededError) as error:     # not signed in / no credits
        raise CloudAccessError(f"answer model {MODELS.answer}: {error}"[:400]) from error
    finally:
        record_answer_usage(run_usage, MODELS.answer)


async def run_search(prompt, request, deps, tool_calls_limit):
    """Run one searching agent. Returns (note, tool calls made, stop reason). Hitting the limit is not a crash:
    the evidence read so far stays in deps.evidence and goes to the answer writer."""
    from pydantic_ai import capture_run_messages
    from pydantic_ai.messages import ToolCallPart
    note, stop_reason = "", "finished"
    with capture_run_messages() as messages:
        try:
            result = await run_counted(
                searching_agent(prompt), request, deps=deps, usage_limits=UsageLimits(tool_calls_limit=tool_calls_limit, request_limit=None))
            note = result.output
        except UsageLimitExceeded:
            stop_reason = f"tool call limit ({tool_calls_limit}) reached"
        except CloudAccessError:
            raise
        except Exception as error:                                 # model/tool failure: keep what was read
            stop_reason = f"{type(error).__name__}: {error}"[:300]
    calls = [f"{part.tool_name}({part.args})" for message in messages for part in message.parts
             if isinstance(part, ToolCallPart)]
    return note, calls, stop_reason


def repair_writer():
    return Agent(model(), instructions=REPAIR_PROMPT, model_settings=settings(), retries=1)


def polish_writer():
    return Agent(model(), instructions=POLISH_PROMPT, model_settings=settings(), retries=1)


def split_writer():
    return Agent(model(), instructions=SPLIT_PROMPT, model_settings=settings(), retries=1)


def about_writer():
    return Agent(model(), instructions=ABOUT_PROMPT, model_settings=settings(), retries=1)


def general_writer():
    return Agent(model(), instructions=prompt(GENERAL_PROMPT), model_settings=settings(), retries=1)


def character_writer():
    """The feel of a dance from own knowledge (CharacterPath), shown under its own heading, never as the books."""
    return Agent(model(), instructions=prompt(CHARACTER_GENERAL_PROMPT), model_settings=settings(), retries=1)


def answer_writer():
    return Agent(model(), instructions=prompt(ANSWER_PROMPT), model_settings=settings(), retries=1)
