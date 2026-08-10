# Tetris

Build a complete Tetris game in a single HTML file with embedded CSS and JavaScript.

**Requirements:**

- **Board:** 10 columns × 20 rows, each cell 30px.
- **All 7 tetrominoes:** I, O, T, S, Z, J, L with standard colors.
- **Rotation:** Wall kicks — if a rotation would cause a collision, try shifting the piece left/right/up to fit (SRS-style basic kicks).
- **Ghost piece:** Show a translucent outline of where the current piece will land.
- **Hard drop:** Press Space to instantly drop the piece to the bottom.
- **Soft drop:** Press Down to accelerate the fall.
- **Line clearing:** When a row is full, clear it and drop all rows above. Score: 1 line = 100, 2 = 300, 3 = 500, 4 = 800.
- **Next piece preview:** Show the next tetromino in a side panel.
- **Hold piece:** Press C to hold/swapping the current piece (one hold slot, can only hold once per drop).
- **Leveling:** Every 10 lines cleared increases the level and speeds up the drop interval.
- **Game over:** When a new piece cannot be placed (collision at spawn).

**Constraints:**

- Single HTML file, no external dependencies.
- Use HTML5 Canvas for rendering.
- Drop interval starts at 1000ms and decreases by 50ms per level (minimum 100ms).
- No `alert()` or `confirm()` — use inline UI for game over / restart.
