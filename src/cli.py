"""Interactive CLI

    python -m src.cli
"""
from __future__ import annotations

from src.graph import ask


def main() -> None:
    print("Rigshospitalet Patient Info Assistant — type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break

        state = ask(q)
        print(f"\nAssistant ({state.get('intent', '?')}):\n{state.get('answer', '')}")
        if state.get("citations"):
            print(f"\n[Sources: {', '.join(state['citations'])}]")
        print()


if __name__ == "__main__":
    main()
