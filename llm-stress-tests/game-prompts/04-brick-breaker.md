# Test 4: "Brick Breaker" Canvas Physics Engine

Create a single-file HTML application featuring a functional Arcade Brick Breaker (Breakout) game using the HTML5 Canvas API. The file must contain all HTML, CSS, and vanilla JS without external libraries or missing code.

Requirements:
1. Canvas Loop: Implement a game rendering loop using `requestAnimationFrame`.
2. Controls: Smooth paddle movement controllable via Arrow Keys or Mouse movement.
3. Physics & Collisions:
   - Ball movement with precise bounding-box collision logic for the canvas walls, paddle, and brick grid.
   - Dynamic Paddle Bounce Angle: The angle at which the ball reflects off the paddle must vary based on where it impacts (e.g., hitting the far left/right edges of the paddle produces a sharper horizontal angle).
4. Brick Array: Render a grid of colored bricks. Destroy a brick upon impact and reverse ball trajectory appropriately.
5. Game State: Track score, lives (3 lives), Win condition (all bricks cleared), Game Over condition (ball drops below paddle), and a "Restart Game" interface.
