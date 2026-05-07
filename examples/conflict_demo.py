from app.builder import build_memory
from examples.support.conflict_demo_tools import demo_conflict_answer, render_debug


mem = build_memory(use_llm=False, reject_conflicts=False)

mem.write_fact("Alice", "location", "lighthouse")
mem.write_fact("Alice", "location", "bridge")
mem.write_fact("Bob", "works_with", "Alice")

query = "Where is Alice?"
bundle = mem.retrieve(query)
conflicts = mem.consistency.detect_conflicts(bundle.facts)
answer = demo_conflict_answer(bundle, conflicts)

print(render_debug(bundle, conflicts, answer))
print("\nFinal answer:")
print(answer)
