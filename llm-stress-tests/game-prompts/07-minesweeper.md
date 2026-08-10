# Minesweeper

Build a complete Minesweeper game in a single HTML file with embedded CSS and JavaScript.

**Requirements:**

- **Grid:** 16×16 with 40 mines (intermediate difficulty).
- **First-click safety:** The first click must never be a mine. Generate the mine field after the first click, excluding the clicked cell and its neighbors.
- **Recursive flood-fill:** Clicking an empty cell (0 adjacent mines) automatically reveals all connected empty cells and their numbered borders.
- **Right-click flagging:** Right-click to flag/unflag a cell. Flagged cells cannot be left-clicked.
- **Number coloring:** Standard Minesweeper colors (1=blue, 2=green, 3=red, 4=dark blue, 5=maroon, 6=teal, 7=black, 8=gray).
- **Timer:** Starts on first click, stops on game over or win.
- **Mine counter:** Shows remaining mines (total mines − flags placed).
- **Win detection:** Game wins when all non-mine cells are revealed.
- **Game over:** Clicking a mine reveals all mines, highlights the clicked mine in red, and marks incorrectly placed flags.
- **Smiley button:** Click to restart the game.

**Constraints:**

- Single HTML file, no external dependencies.
- Use CSS Grid for the board layout.
- No `alert()` or `confirm()` — use inline UI elements.
- Responsive: board should fit on a standard laptop screen without scrolling.
