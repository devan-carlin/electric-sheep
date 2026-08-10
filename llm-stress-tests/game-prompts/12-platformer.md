# Simple Platformer

Build a simple 2D platformer game in a single HTML file with embedded CSS and JavaScript.

**Requirements:**

- **Player:** A character that can move left/right and jump. Gravity pulls the player down when not on a platform.
- **Platforms:** At least 8 platforms of varying widths and heights, including:
  - A ground platform (full width).
  - Floating platforms at different heights.
  - A moving platform (oscillates left/right).
- **Collectibles:** 10 coins scattered across platforms. Collecting a coin increments the score and removes the coin.
- **Hazards:** 3 spike traps on the ground or platforms. Touching a spike resets the player to the start position.
- **Camera:** The view scrolls to follow the player horizontally (level is wider than the viewport).
- **Win condition:** Reach a flag/goal at the far right of the level.
- **HUD:** Shows score (coins collected) and deaths (spike hits).
- **Controls:** Arrow keys or WASD for movement, Space or Up for jump.

**Constraints:**

- Single HTML file, no external dependencies.
- Use HTML5 Canvas for rendering.
- Collision detection must be precise (player stands on top of platforms, doesn't fall through).
- Double jump is not allowed — player can only jump when on a platform (or in the air for a single jump).
- No `alert()` or `confirm()` — use inline UI for win screen.
