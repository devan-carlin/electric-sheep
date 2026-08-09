# Test 2: Constraint Satisfaction — The Complex Shift Scheduler

**Target Capability:** Multi-variable logic and negative constraints ("Do NOT...").

**Why it's a great test:** LLMs excel at language fluency but naturally struggle with constraint satisfaction problems (CSPs), especially when negative rules are involved.

**The Task:** Give the model 6 employees, 7 days of the week, and 3 shifts per day. Give it a set of conflicting rules. Ask it to produce a complete schedule as a markdown table. Once it generates one, feed the schedule back into a fresh instance and ask: *"Verify if this schedule breaks any of the rules."* (Tests self-auditing capabilities).

---

## Prompt

```
You are a shift scheduling engine. Create a complete work schedule for the following 6 employees across 7 days (Monday through Sunday) with 3 shifts per day: Morning (06:00–14:00), Afternoon (14:00–22:00), Night (22:00–06:00).

Each shift must have exactly 1 employee assigned. That means 21 total shift slots per week.

Employees: Alice, Bob, Charlie, Dave, Eve, Frank

Hard Constraints (ALL must be satisfied):
1. Alice can ONLY work Morning shifts.
2. Bob cannot work on consecutive days (he must have at least 1 day off between any two work days).
3. Charlie and Dave cannot work the same shift on the same day.
4. Eve must have at least 2 full days off (she works at most 5 days per week).
5. Frank can ONLY work Night shifts.
6. No employee can work more than 5 shifts per week.
7. Every shift slot must be filled (no gaps).

Output the schedule as a Markdown table with columns: Day | Morning | Afternoon | Night

After generating the schedule, verify each constraint and list any violations. If violations exist, fix them and output the corrected schedule.
```
