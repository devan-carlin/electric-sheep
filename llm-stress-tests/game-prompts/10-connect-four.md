# Connect Four

Build a complete Connect Four game in a single HTML file with embedded CSS and JavaScript.

**Requirements:**

- **Board:** 7 columns × 6 rows.
- **Two-player mode:** Human vs. human, alternating red and yellow discs.
- **AI opponent:** Optional toggle for human vs. computer. AI uses minimax with alpha-beta pruning (depth ≥ 6).
- **Gravity:** Discs fall to the lowest available row in the selected column.
- **Win detection:** Check for 4-in-a-row horizontally, vertically, and both diagonals after each move.
- **Draw detection:** Board full with no winner.
- **Hover preview:** Show a ghost disc in the selected column before dropping.
- **Winning line highlight:** Draw a line or highlight the 4 winning discs.
- **Score tracking:** Track wins/losses/draws for both players (persisted in localStorage).
- **Reset button:** Clear the board and start a new game.

**Constraints:**

- Single HTML file, no external dependencies.
- Use CSS Grid or Canvas for the board.
- AI should have a noticeable but short thinking time (< 500ms) — use iterative deepening or a move ordering heuristic if needed.
- No `alert()` or `confirm()` — use inline UI for win/draw notifications.
