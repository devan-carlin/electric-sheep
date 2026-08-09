
# Test 6: Sudoku Generator, Solver & Interactive Playable Grid

Create a single, complete HTML file containing all HTML, CSS, and JavaScript for a fully playable Sudoku application. Do not use external frameworks, libraries, or placeholder comments.

Requirements:

1. **Board Generation**: Implement a backtracking algorithm that generates a valid, complete 9x9 Sudoku solution, then removes a configurable number of cells (Easy: 30 removed, Medium: 40 removed, Hard: 50 removed, Expert: 55 removed) to create the puzzle. The resulting puzzle must have a unique solution.
2. **Solver**: Implement a backtracking solver that can solve any valid 9x9 Sudoku board. Display the solution when the user clicks "Solve" or "Show Solution."
3. **Interactive Grid**: 
   - Render a 9x9 grid with clear visual distinction between the three 3x3 sub-grids (thicker borders).
   - Pre-filled cells are visually distinct (e.g., bold, darker color) from user-filled cells.
   - Allow the user to select a cell (click) and enter a digit (1–9 via keyboard or an on-screen number pad).
   - Allow the user to clear a cell (press Delete, Backspace, or 0).
4. **Validation & Highlighting**:
   - In real time, highlight conflicting cells (same row, column, or 3x3 box containing duplicate digits) in red.
   - Highlight the currently selected cell's row, column, and 3x3 box with a subtle background color.
   - Highlight all cells containing the same digit as the currently selected cell.
5. **Game State**:
   - Track elapsed time with a visible timer (MM:SS).
   - Detect and announce a win condition when the board is correctly and completely filled.
   - Provide a "New Game" button with difficulty selection.
   - Provide an "Undo" button that reverts the last user move (maintain a move history stack).
6. **Notes/Pencil Mode**: Toggle a notes mode where entering a digit adds it as a small pencil mark in the cell (multiple notes per cell). Clicking a note promotes it to the cell value. Notes are displayed in a 3x3 mini-grid within the cell.
7. **Styling**: Clean, modern CSS with responsive layout that works on both desktop and mobile. Use CSS Grid for the board layout. Include smooth transitions for cell selection and hover states.

