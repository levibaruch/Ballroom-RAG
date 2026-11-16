"""Which installed Ollama models fit which role, and which ones are selected now.

    python -m dance_rag_agentic.agent.models

Only asks Ollama for its model list and each model's capabilities (/api/tags, /api/show): no model is loaded.
    python -m dance_rag_agentic.agent.models --warm     # load both selected models now and keep them loaded
"""
import argparse
import os

import requests

from dance_rag_agentic.config import API_PROVIDERS, CLOUD_MODELS, KEEP_ALIVE, MODELS, OLLAMA_URL, SETTINGS_PATH, api_provider

ROLE_NEEDS = {"answer": {"completion", "tools"}, "decision": {"decision"}}


def installed_models():
    """[{'name', 'parameters', 'capabilities': set}] for every model Ollama has."""
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    models = []
    for tag in tags:
        details = requests.post(f"{OLLAMA_URL}/api/show", json={"model": tag["name"]}, timeout=10).json()
        models.append(dict(name=tag["name"], parameters=tag.get("details", {}).get("parameter_size", "?"),
                           capabilities=set(details.get("capabilities", []))))
    return models


CLOUD_DECISION_MIN_VERSION = (0, 40, 5)       # clef:cloud through the local server needs this Ollama version


def is_cloud(name):
    return any(_same_model(name, cloud) for cloud_list in CLOUD_MODELS.values() for cloud in cloud_list)


def ollama_version():
    text = requests.get(f"{OLLAMA_URL}/api/version", timeout=10).json().get("version", "0")
    return tuple(int(part) for part in text.split("-")[0].split(".") if part.isdigit())


def pull_cloud_model(name):
    """A cloud model is a small stub in Ollama (no weights); pull it once so Ollama can route to ollama.com.
    Returns a problem text or None."""
    try:
        response = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": name, "stream": False}, timeout=120)
        if response.status_code != 200:
            return f"cannot pull cloud model '{name}': {response.text[:200]} (signed in? run: ollama signin)"
    except requests.RequestException as error:
        return f"cannot pull cloud model '{name}': {type(error).__name__}"
    return None


def _same_model(name, other):
    """'autotrust-jev' and 'autotrust-jev:latest' are the same model."""
    def full(model_name):
        return model_name if ":" in model_name else f"{model_name}:latest"
    return full(name) == full(other)


def role_problems():
    """Problems with the current selection ([] = fine): model not installed, or missing a capability.
    A cloud model from settings.toml [cloud] that is not pulled yet is pulled here (a small stub, no weights)."""
    try:
        models = installed_models()
    except requests.RequestException as error:
        return [f"cannot reach Ollama at {OLLAMA_URL}: {type(error).__name__}"]
    problems = []
    for role, needs in ROLE_NEEDS.items():
        chosen = getattr(MODELS, role)
        provider_name, _ = api_provider(chosen)
        if provider_name:                               # runs on that provider's API, not on Ollama
            if role != "answer":
                problems.append(f"{role} model '{chosen}': only the answer model can run on {provider_name} "
                                f"(the decision model needs Ollama)")
            elif not os.environ.get(API_PROVIDERS[provider_name]):
                problems.append(f"answer model '{chosen}' needs {API_PROVIDERS[provider_name]} (in deploy/.env or the environment)")
            continue
        match = next((model for model in models if _same_model(model["name"], chosen)), None)
        if match is None and is_cloud(chosen):          # cloud model not pulled yet: pull its stub now
            problem = pull_cloud_model(chosen)
            if problem:
                problems.append(problem)
                continue
            models = installed_models()
            match = next((model for model in models if _same_model(model["name"], chosen)), None)
        if role == "decision" and is_cloud(chosen) and ollama_version() < CLOUD_DECISION_MIN_VERSION:
            problems.append(f"cloud decision model '{chosen}' needs Ollama "
                            f"{'.'.join(map(str, CLOUD_DECISION_MIN_VERSION))} or newer")
        if match is None:
            problems.append(f"{role} model '{chosen}' is not installed (ollama pull {chosen})")
        elif is_cloud(chosen) and not match["capabilities"]:
            continue                                    # a stub may not list capabilities: settings.toml decides
        elif not needs <= match["capabilities"]:
            problems.append(f"{role} model '{chosen}' lacks {sorted(needs - match['capabilities'])}; "
                            f"it has {sorted(match['capabilities'])}")
    return problems


def add_model_flags(parser):
    """--answer-model / --decision-model for every command line."""
    parser.add_argument("--answer-model", help=f"Ollama model for agents + answer (now: {MODELS.answer})")
    parser.add_argument("--decision-model", help=f"Ollama decision model (now: {MODELS.decision})")


def warm_up():
    """Load both selected models now and keep them loaded for KEEP_ALIVE (no answer is generated)."""
    if not api_provider(MODELS.answer)[0]:              # an API model needs no loading
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": MODELS.answer, "keep_alive": KEEP_ALIVE},
                      timeout=600).raise_for_status()             # no prompt = load only
    requests.post(f"{OLLAMA_URL}/v1/systemone", timeout=600, json={
        "model": MODELS.decision, "state": "warm up", "keep_alive": KEEP_ALIVE,
        "questions": {"ready": {"type": "noul", "instructions": "Is this text a greeting?"}}}).raise_for_status()
    loaded = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    for model in loaded:
        print(f"loaded {model['name']:28} {model.get('size', 0) / 1e9:5.1f} GB  until {model.get('expires_at', '?')[:19]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--warm", action="store_true", help=f"load both selected models now, keep them {KEEP_ALIVE}")
    add_model_flags(parser)
    arguments = parser.parse_args()
    from dance_rag_agentic.config import use_models
    use_models(arguments.answer_model, arguments.decision_model)
    if arguments.warm:
        warm_up()
        return
    models = installed_models()
    print(f"Selected (settings: {SETTINGS_PATH.name}, env vars and CLI flags override):")
    print(f"  answer   = {MODELS.answer}\n  decision = {MODELS.decision}\n")
    print(f"{'model':28} {'size':>7}  {'answer':6}  {'decision':8}  capabilities")
    for model in sorted(models, key=lambda model: model["name"]):
        fits = {role: needs <= model["capabilities"] for role, needs in ROLE_NEEDS.items()}
        selected = [role for role in ROLE_NEEDS if _same_model(model["name"], getattr(MODELS, role))]
        marks = {role: ("*" if role in selected else "yes") if fits[role] else "-" for role in ROLE_NEEDS}
        print(f"{model['name']:28} {model['parameters']:>7}  {marks['answer']:6}  {marks['decision']:8}  "
              f"{', '.join(sorted(model['capabilities']))}")
    print("\n* = selected now.  Switch for one run: --answer-model NAME / --decision-model NAME")
    for problem in role_problems():
        print(f"PROBLEM: {problem}")


if __name__ == "__main__":
    main()
