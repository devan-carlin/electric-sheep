# Output: unsloth-qwen3.6-27b × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Response
Here's a thinking process:

1.  **Analyze User Input:**
   - **Topic:** Algorithm Design
   - **Format Required:** Approach/Algorithm Name, Time/Space Complexity, Brief 1-2 sentence explanation for each of 3 problems.
   - **Problem 1:** Sliding Window Median (array of n integers, window size k, return median for each window). Target: O(n log k). Question: What data structure?
   - **Problem 2:** Articulation Points (find vertices whose removal disconnects undirected graph). Target: O(V+E) single DFS. Question: What are the key conditions (back-edge vs tree-edge)?
   - **Problem 3:** Max Overlapping Intervals (n intervals, find max overlaps at any point). Target: O(n log n). Question: What sweep-line technique?

2.  **Deconstruct & Solve Each Problem:**

   **Problem 1: Sliding Window Median**
   - *Goal:* Maintain median of a sliding window of size k over an array of size n.
   - *Target Complexity:* O(n log k) time, O(k) space.
   - *Data Structure:* Two heaps (max-heap for lower half, min-heap for upper half) or a balanced BST/multiset. In practice, two heaps with lazy deletion or a balanced BST (like `std::multiset` in C++ or `SortedList` in Python) works. The classic approach for O(n log k) uses two heaps with lazy removal or a balanced BST.
   - *Approach Name:* Two-Heap / Balanced BST approach (often called "Running Median" or "Sliding Window Median with Heaps/BST").
   - *Complexity:* Time: O(n log k), Space: O(k).
   - *Explanation:* Maintain two heaps (or a balanced BST) to split the window into lower and upper halves, allowing O(log k) insertion/deletion and O(1) median access. As the window slides, remove the outgoing element and add the incoming one, rebalancing to keep sizes equal or off-by-one.

   **Problem 2: Articulation Points**
   - *Goal:* Find articulation points in an undirected graph.
   - *Target Complexity:* O(V+E) time, O(V) space (for recursion/arrays).
   - *Algorithm Name:* Tarjan's Algorithm /

## Metadata
- **Model**: unsloth-qwen3.6-27b
- **Elapsed**: 16306ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
- **Gen tok/s**: 31.3
