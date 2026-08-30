# Output: top-32 × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Response
Here's a thinking process:

1.  **Analyze User Request:**
   - The user wants algorithm design for three specific problems.
   - For each, they need:
     - Algorithm name/approach
     - Time/space complexity
     - Brief 1-2 sentence explanation
   - Constraints/Targets are specified:
     1. Sliding Window Median: O(n log k)
     2. Articulation Points: O(V+E) single DFS
     3. Max Overlapping Intervals: O(n log n) sweep-line
   - They also ask specific questions about data structures/key conditions/techniques.

2.  **Problem 1: Sliding Window Median**
   - Target: O(n log k)
   - Need a data structure that supports efficient insertion, deletion, and median retrieval for a window of size k.
   - Common approaches: Two heaps (min/max), balanced BST, or specialized structures like `MedianFinder` with two heaps.
   - Two heaps approach: Maintain a max heap for the left half and a min heap for the right half. When sliding, remove the outgoing element, add the incoming element, rebalance heaps. Median is top of max heap (or average of tops if even size).
   - Complexity: Each slide involves O(log k) operations for heap insert/delete. Total O(n log k). Space O(k).
   - Algorithm name: Two-Heap Sliding Window / Balanced Heap Approach.
   - Explanation: Maintain two heaps (max and min) to split the window into halves; rebalance after each slide to keep median at the top of the max heap.
   - Check constraints: Matches O(n log k), space O(k).

3.  **Problem 2: Articulation Points**
   - Target: O(V+E) single DFS
   - Key conditions: back-edge vs tree-edge, discovery time, low value.
   - Algorithm: DFS-based Articulation Point Detection (Tarjan's algorithm or standard DFS approach).
   - Key conditions: For vertex u, if it has a child v in DFS tree such that `low[v] >= disc[u]`, then u is an articulation point. Also, root is articulation point if it has >1 children in DFS tree.
   - Explanation: During DFS, track discovery times and lowest reachable discovery times via back edges; a vertex is an articulation point if any subtree

## Metadata
- **Top-k**: 32
- **Elapsed**: 7004ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
