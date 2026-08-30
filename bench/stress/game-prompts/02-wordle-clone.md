# Test 2: Strict Logic "Wordle" Clone

Create a complete, fully functional, single-file HTML application that clones Wordle. Place all HTML, CSS, and JavaScript inside this single file. Do not use external dependencies or standard placeholders.

Requirements:
1. Board & Input: Render a 6-row by 5-column letter grid and a modern on-screen virtual keyboard. Allow both physical keyboard typing and virtual keyboard clicks.
2. Word Bank: Hardcode an internal array of at least 20 valid 5-letter target words and choose one randomly per game.
3. Exact Color Evaluation Logic (Crucial):
   - Correct letter & position = Green.
   - Correct letter, wrong position = Yellow.
   - Letter not in word = Gray.
   - Strict Duplicate Rule: If a letter is guessed multiple times in one attempt but appears fewer times in the target word, color priority must strictly go to Green first, then Yellow (from left to right up to the count in the word), and remaining instances MUST stay Gray.
4. UI State: Update the colors of the on-screen virtual keyboard to match the highest status revealed so far (Green > Yellow > Gray).
5. Animations & End State: Flip animation on tile reveals, win/loss modal, and a "Play Again" button that resets state cleanly.
