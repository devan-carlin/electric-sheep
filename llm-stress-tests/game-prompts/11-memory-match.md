# Memory Card Match

Build a complete Memory Card Match game in a single HTML file with embedded CSS and JavaScript.

**Requirements:**

- **Grid:** 4×6 (24 cards, 12 unique pairs).
- **Card flip animation:** Smooth 3D CSS flip on click (front shows a symbol/icon, back shows a uniform design).
- **Matching logic:** Flip two cards at a time. If they match, keep them face-up. If not, flip them back after 1 second.
- **Shuffle:** Fisher-Yates shuffle at game start.
- **Move counter:** Track the number of pairs attempted (each pair of flips = 1 move).
- **Timer:** Starts on first card flip, stops when all pairs are found.
- **Star rating:** 3 stars if moves ≤ 18, 2 stars if moves ≤ 24, 1 star if moves > 24.
- **Win screen:** Shows time, moves, and star rating with a "Play Again" button.
- **Icons:** Use Unicode emoji or SVG icons (no external image dependencies).

**Constraints:**

- Single HTML file, no external dependencies.
- Cards should be clickable only when both currently flipped cards have been resolved (matched or flipped back).
- Responsive: board should fit on a standard laptop screen without scrolling.
- No `alert()` or `confirm()` — use inline UI for the win screen.
