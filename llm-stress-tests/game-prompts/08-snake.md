# Snake

Build a complete Snake game in a single HTML file with embedded CSS and JavaScript.

**Requirements:**

- **Grid:** 20×20 cells, each cell 25px (500×500 canvas).
- **Movement:** Arrow keys or WASD to change direction. Snake moves continuously in the current direction at a fixed interval (~150ms per tick).
- **Growth:** Eating food grows the snake by one segment and increases the score.
- **Food spawning:** New food appears at a random empty cell (not on the snake's body).
- **Collision detection:** Game over if the snake hits the wall or its own body.
- **Score display:** Current score and high score (persisted in localStorage).
- **Speed increase:** Every 5 food items eaten, reduce the tick interval by 10ms (minimum 50ms).
- **Pause:** Press Space to pause/resume.
- **Game over screen:** Shows final score, high score, and a "Play Again" button.

**Constraints:**

- Single HTML file, no external dependencies.
- Use HTML5 Canvas for rendering.
- Smooth rendering — no flickering or double-drawing artifacts.
- Prevent the snake from reversing into itself (e.g., if moving right, pressing left is ignored).
