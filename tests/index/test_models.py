"""Index tests (no model; need the built index): model settings, prompts, the token meter.
Each test pins down a bug seen in a real run."""


def test_prompts_contain_no_test_cases():
    # examples in prompts must be abstract: a prompt that quotes a test case makes that test worthless
    from dance_rag_agentic.agent import agents, prompts as prompt_texts
    prompts = [agents.prompt(template) for template in (prompt_texts.TECHNIQUE_PROMPT, prompt_texts.UNSURE_PROMPT,
                                                        prompt_texts.FILLGAP_PROMPT, prompt_texts.ANSWER_PROMPT,
                                                        prompt_texts.BACKGROUND_PROMPT, prompt_texts.GENERAL_PROMPT,
                                                        prompt_texts.CHARACTER_PROMPT, prompt_texts.CHARACTER_GENERAL_PROMPT)]
    prompts += [prompt_texts.SPLIT_PROMPT, prompt_texts.MIXED_INSTRUCTIONS, *prompt_texts.MIXED_CRITERIA.values()]
    prompts += [prompt_texts.CLAIM_INSTRUCTIONS, *prompt_texts.CLAIM_CRITERIA.values(),
                prompt_texts.ANSWERS_QUESTION_INSTRUCTIONS, *prompt_texts.ANSWERS_QUESTION_CRITERIA.values()]
    from dance_rag_agentic.agent.tools import TOOLBOX
    prompts += [tool.__doc__ for tool in TOOLBOX]                       # tool docstrings are prompts too
    leaks = ["3/4", "4/4", "time signature", "Fan Position", "Contrary Body", "counter body", "Hockey Stick",
             "CBM", "upper body against", "Music in DanceSport", "\"fan\""]
    found = [(leak, text[:60]) for text in prompts for leak in leaks if leak.lower() in text.lower()]
    assert found == []
