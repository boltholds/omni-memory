from app.memory import OmniMemory

mem = OmniMemory()

mem.write_fact("Alice", "location", "lighthouse")
mem.write_fact("Alice", "location", "bridge")
mem.write_fact("Bob", "works_with", "Alice")

answer = mem.ask("Where is Alice?", debug=True)

print("\nFinal answer:")
print(answer.answer)