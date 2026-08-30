# Logic Puzzle — Constraint Satisfaction

Solve the following puzzle. Show your reasoning process (elimination table or step-by-step deduction).

---

## The Five Houses Puzzle

There are five houses in a row, each with a different color. Each house is occupied by a person of a different nationality. Each person drinks a different beverage, smokes a different brand, and keeps a different pet.

**Clues:**

1. The Brit lives in the red house.
2. The Swede keeps dogs.
3. The Dane drinks tea.
4. The green house is immediately to the left of the white house.
5. The green house's owner drinks coffee.
6. The person who smokes Pall Mall keeps birds.
7. The yellow house's owner smokes Dunhill.
8. The person living in the center house drinks milk.
9. The Norwegian lives in the first house.
10. The person who smokes Blends lives next to the person who keeps cats.
11. The person who keeps horses lives next to the person who smokes Dunhill.
12. The person who smokes Blue Master drinks beer.
13. The German smokes Prince.
14. The Norwegian lives next to the blue house.
15. The person who smokes Blends has a neighbor who drinks water.

**Questions:**

1. Who owns the fish?
2. What does the Norwegian drink?
3. What color is the German's house?
4. Who smokes Blends?

---

## Second Puzzle: Scheduling

Four developers (Alice, Bob, Carol, Dave) need to be assigned to four projects (Alpha, Beta, Gamma, Delta). Each developer works on exactly one project, and each project has exactly one developer.

**Constraints:**

- Alice cannot work on Alpha (she's the former lead and it's a conflict of interest).
- Bob must work on Beta or Gamma (those are his expertise areas).
- Carol refuses to work on Delta (she's already burnt out from the previous cycle).
- Dave cannot work on Gamma (he's on paternity leave during the critical phase).
- If Bob works on Beta, then Alice must work on Gamma.
- Carol and Alice cannot work on the same project (obvious, but stated for completeness).

**Questions:**

1. List all valid assignments.
2. If we add the constraint "Dave must work on Alpha or Delta," how many valid assignments remain?
3. Is there a unique solution if Bob must work on Gamma?

---

**Scoring criteria:**

| Criterion | Points |
|-----------|--------|
| Correct final answers | 50% |
| Clear deduction process shown | 30% |
| No contradictions in reasoning | 10% |
| Handles all sub-questions | 10% |
