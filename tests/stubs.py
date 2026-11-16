"""Stand-ins for the decision model, shared by unit and index tests (no index, no model)."""

KINDS = ["FIGURE", "FIGURE_COMPARE", "TECHNIQUE", "TERM", "GENERAL", "OUT_OF_SCOPE"]


def probabilities(best, p_best=0.8):
    """Decision-model-like probabilities with one clear winner."""
    rest = (1 - p_best) / (len(KINDS) - 1)
    return {kind: (p_best if kind == best else rest) for kind in KINDS}


def classify_stub(kind, mixed):
    def classify(question):
        probabilities = {name: (0.8 if name == kind else 0.04) for name in KINDS}
        return {"kind": {"choice": kind, "probabilities": probabilities, "confidence": 0.7},
                "aspect": {"choice": "all", "probabilities": {"all": 0.9}}, "mixed": {"noul": mixed}}
    return classify
