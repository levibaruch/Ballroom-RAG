"""Index tests (no model; need the built index): entities (dances, figures, roles, steps) and the routing rules.
Each test pins down a bug seen in a real run."""
import pytest
from dance_rag_agentic.agent.entities import extract_entities
from dance_rag_agentic.agent.graph import choose_route
from .shared import probabilities


@pytest.mark.parametrize("question, figures", [
    ("hip design on step 3 of the openin out, man", ["Opening Out"]),       # typo in figure name
    ("What is the man's hip design on step 3 of the Hockey Stick?", ["Hockey Stick"]),   # 'hip' is not 'Whip'
    ("What is the timing of the Lindy Hop swing out?", []),                 # 'swing' is not 'Wing'
    ("What is Fan Position in Jive?", []),                                  # a position, not the figure Fan
    ("How does the Whip differ from the Throwaway Whip in Jive?", ["Throwaway Whip", "Whip"]),
    ("Footwork of man and lady in the Spot Turn to Right", ["Spot Turn To Right"]),   # not 'Three Step Turn'
    ("wisk step 3 man quickstep, what foot action and timing??", ["Whisk"]),       # typo in a short name
    ("Welke voetplaatsing heeft de heer op stap 2 van de Samba Whisk naar links?", ["Whisk"]),   # 'links' is not Link
    ("what are the three types of sway in slow waltz", []),                      # not 'Three Threes'
    ("whats the wing postion in viennese waltz", []),                             # typo in 'position'
])
def test_figures_found(question, figures):
    assert sorted(extract_entities(question)["figures"]) == sorted(figures)


def test_dances_role_and_steps():
    entities = extract_entities("How does the lady's Fan differ between Rumba and cha cha on step 2?")
    assert sorted(entities["dances"]) == ["Cha Cha", "Rumba"]
    assert entities["role"] == "lady"
    assert entities["steps"] == [2]
    assert extract_entities("man and lady in the Fan")["role"] == "both"


def test_rule_1_no_figure_means_no_figure_route():
    # 'How does the man lead a turn?' -> the decision model said FIGURE 0.57; no figure is named
    route, _ = choose_route(probabilities("FIGURE"), 0.6, extract_entities("How does the man lead a turn?"))
    assert route != "FIGURE"


def test_rule_2_two_dances_is_a_comparison():
    entities = extract_entities("How does the lady's Fan differ between Rumba and Cha Cha?")
    route, _ = choose_route(probabilities("TECHNIQUE"), 0.9, entities)
    assert route == "FIGURE_COMPARE"


def test_rule_3_low_confidence_goes_to_unsure():
    route, _ = choose_route(probabilities("TERM", p_best=0.35), 0.2, extract_entities("What are swivel actions?"))
    assert route == "UNSURE"


def test_confident_model_choice_is_kept():
    route, reason = choose_route(probabilities("TERM"), 0.7, extract_entities("What does LOD stand for?"))
    assert (route, reason) == ("TERM", "decision model")


def test_partial_figure_names_are_hinted_not_entities():
    from dance_rag_agentic.agent.entities import extract_entities, figures_ending_in_question_word
    question = "How do I do samba runs?"
    assert extract_entities(question)["figures"] == []                 # two candidates would mean COMPARE
    assert figures_ending_in_question_word(question, ["Samba"]) == [
        "Carioca Runs (Samba)", "Promenade To Counter Promenade Runs (Samba)"]
