"""Command line:  python -m dance_rag_agentic.apps.cli "How do I dance the Fan in Jive?" [--trace] [--no-judge]"""
import argparse

import time

from dance_rag_agentic.config import MODELS, TRACE_DIR, use_models
from dance_rag_agentic.agent.graph import ask
from dance_rag_agentic.agent.models import add_model_flags, role_problems


def main():
    parser = argparse.ArgumentParser(description="Answer a dance technique question from the WDSF books.")
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="print every node and tool call while working")
    parser.add_argument("--no-judge", action="store_true", help="skip JudgeClaims (faster)")
    add_model_flags(parser)
    arguments = parser.parse_args()
    use_models(arguments.answer_model, arguments.decision_model)
    problems = role_problems()
    if problems:
        raise SystemExit("cannot run:\n- " + "\n- ".join(problems))
    if arguments.trace:
        print(f"models: answer={MODELS.answer}  decision={MODELS.decision}")
    trace_path = None
    if arguments.trace:                     # the log is written line by line while the graph runs
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.log"
        number = 2
        while trace_path.exists():          # parallel runs in the same second wrote into one file
            trace_path = TRACE_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{number}.log"
            number += 1
        trace_path.write_text(f"QUESTION: {arguments.question}\n"
                              f"MODELS: answer={MODELS.answer} decision={MODELS.decision}\n", encoding="utf-8")
        print(f"trace log: {trace_path}   (follow it with: tail -f '{trace_path}')")
    state = ask(arguments.question, use_judge=not arguments.no_judge, live=arguments.trace, log_path=trace_path)
    final_block = ("\n" + "═" * 74 + f"\nFINAL ANSWER  (route {state.route}: {state.route_reason})\n" + "═" * 74
                   + "\n" + state.final_answer)
    print(final_block if arguments.trace else state.final_answer)
    if trace_path:
        with open(trace_path, "a", encoding="utf-8") as log_file:
            log_file.write(final_block + "\n")
        print(f"\ntrace saved: {trace_path}")


if __name__ == "__main__":
    main()
