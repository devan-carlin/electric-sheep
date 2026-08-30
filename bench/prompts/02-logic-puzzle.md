# Logic Puzzle — Constraint Satisfaction

**Zebra Puzzle (classic):** Five houses, each with a unique color, nationality, drink, smoke brand, and pet. Clues:
1. Brit → red house. 2. Swede → dogs. 3. Dane → tea. 4. Green house immediately left of white. 5. Green → coffee. 6. Pall Mall → birds. 7. Yellow → Dunhill. 8. Center house → milk. 9. Norwegian → first house. 10. Blends smoker next to cat owner. 11. Horse owner next to Dunhill smoker. 12. Blue Master → beer. 13. German → Prince. 14. Norwegian next to blue house. 15. Blends smoker next to water drinker.

Answer: Who owns the fish? What does the Norwegian drink? What color is the German's house?

**Scheduling:** Assign 4 devs (Alice, Bob, Carol, Dave) to 4 projects (Alpha, Beta, Gamma, Delta). Constraints: Alice ≠ Alpha. Bob ∈ {Beta, Gamma}. Carol ≠ Delta. Dave ≠ Gamma. If Bob→Beta then Alice→Gamma. List all valid assignments. If we add "Dave ∈ {Alpha, Delta}", how many remain?
