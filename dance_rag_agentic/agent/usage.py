"""Token meter: every model call adds the tokens it used; the trace shows them per node and per run.

Where the numbers come from (both reported by the model server, not estimated):
    answer model    pydantic-ai RunUsage (input_tokens, output_tokens) of each agent run
    decision model  the "usage" object of each /v1/systemone response (input_tokens, output_tokens)

Cost: settings.toml [prices] gives USD per 1M tokens per model; a model without a price is shown without cost
(a local model costs nothing per token).

One meter per process (METER). The graph answers one question at a time, so per question:
    METER.reset() at the start, METER.totals() at the end.
"""
from dataclasses import dataclass

from dance_rag_agentic.config import PRICES


@dataclass
class Call:
    role: str              # "answer" or "decision"
    model: str
    input_tokens: int
    output_tokens: int


class TokenMeter:
    def __init__(self):
        self.calls = []                                  # list[Call], in order

    def reset(self):
        self.calls = []

    def add(self, role, model, input_tokens, output_tokens):
        self.calls.append(Call(role, model, int(input_tokens or 0), int(output_tokens or 0)))

    def mark(self):
        """A position in the call list; totals(since=mark) counts only the calls after it (one node)."""
        return len(self.calls)

    def totals(self, since=0):
        """{(role, model): {"calls", "input", "output", "usd" (None if no price)}} for calls[since:]."""
        result = {}
        for call in self.calls[since:]:
            entry = result.setdefault((call.role, call.model), {"calls": 0, "input": 0, "output": 0})
            entry["calls"] += 1
            entry["input"] += call.input_tokens
            entry["output"] += call.output_tokens
        for (_, model), entry in result.items():
            entry["usd"] = cost(model, entry["input"], entry["output"])
        return result

    def summary(self, since=0):
        """One short line for a node's trace footer, e.g. 'answer 12,034 in / 812 out · decision 3,210 in'."""
        parts = []
        for (role, _), entry in self.totals(since).items():
            part = f"{role} {entry['input']:,} in / {entry['output']:,} out"
            parts.append(part + (f" ${entry['usd']:.4f}" if entry["usd"] else ""))
        return " · ".join(parts)

    def report_lines(self):
        """The per-run table for the Finish node."""
        totals = self.totals()
        lines = [f"{'role':9} {'model':28} {'calls':>5} {'input':>9} {'output':>8} {'USD':>9}"]
        for (role, model), entry in totals.items():
            usd = f"{entry['usd']:.4f}" if entry["usd"] is not None else "-"
            lines.append(f"{role:9} {model[:28]:28} {entry['calls']:5} {entry['input']:9,} {entry['output']:8,} {usd:>9}")
        known = [entry["usd"] for entry in totals.values() if entry["usd"] is not None]
        lines.append(f"{'total':9} {'':28} {sum(e['calls'] for e in totals.values()):5} "
                     f"{sum(e['input'] for e in totals.values()):9,} {sum(e['output'] for e in totals.values()):8,} "
                     f"{(f'{sum(known):.4f}' if known else '-'):>9}")
        return lines


def cost(model, input_tokens, output_tokens):
    """USD for these tokens, or None when settings.toml has no price for the model."""
    price = PRICES.get(model)
    if price is None:
        return None
    input_price, output_price = price
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def record_answer_usage(run_usage, model):
    """Add a pydantic-ai RunUsage (also filled when the run stopped on an error or a limit)."""
    METER.add("answer", model, run_usage.input_tokens, run_usage.output_tokens)


def record_decision_usage(response_json, model):
    usage = response_json.get("usage") or {}
    METER.add("decision", model, usage.get("input_tokens"), usage.get("output_tokens"))


METER = TokenMeter()
