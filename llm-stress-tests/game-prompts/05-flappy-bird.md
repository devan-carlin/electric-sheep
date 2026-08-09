# Test 5: "Flappy Bird" Gravity & Procedural Pipe Engine

Create a complete, fully functional, single-file HTML application featuring a Flappy Bird game using the HTML5 Canvas API. Place all HTML, CSS, and JavaScript inside this single file. Do not use external CSS/JS libraries or placeholders.

Requirements:
1. Game Loop & Gravity Physics: Implement a continuous `requestAnimationFrame` loop. The player bird object must experience continuous downward acceleration (gravity) and apply an upward velocity impulse (jump) whenever the Spacebar is pressed or the screen is clicked/tapped.
2. Dynamic Pipe Generation: Periodically spawn vertical pipe obstacles (top and bottom pairs) scrolling smoothly from right to left. The vertical gap size between the top and bottom pipes must remain navigable, but its vertical height position must vary randomly.
3. Collision & Scoring Logic:
   - Implement accurate collision detection between the bird and top/bottom pipes, floor, and ceiling boundaries.
   - Increment score by +1 each time a pipe pair successfully passes behind the bird without collision.
4. Persistence & States: Store high scores using browser `localStorage`. Provide clear visual states for Start Screen, Active Gameplay, and Game Over with an immediate restart mechanism.
5. Visual Polish: Use a clean arcade-style Canvas aesthetic with a dynamic score counter overlay and smooth rotation on the bird relative to its vertical velocity.
