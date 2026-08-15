# Output: top-8 × 03-algorithm-design

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
   - **Requirements:** For each of the 3 problems, provide:
     - Algorithm name/approach
     - Time/Space complexity
     - Brief 1-2 sentence explanation
   - **Specific Questions per problem:**
     1. Sliding Window Median: Target O(n log k). What data structure?
     2. Articulation Points: Target O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
     3. Max Overlapping Intervals: Target O(n log n). What sweep-line technique?

2.  **Deconstruct & Solve Each Problem:**

   **Problem 1: Sliding Window Median**
   - *Target:* O(n log k) time, O(k) space.
   - *Approach:* Two heaps (max-heap for left half, min-heap for right half) with lazy deletion or a balanced BST (like `std::multiset` in C++ or `SortedList` in Python). The two-heap approach is classic and fits O(n log k).
   - *Data Structure:* Two heaps (max-heap + min-heap) or a balanced BST/multiset.
   - *Complexity:* Time: O(n log k), Space: O(k).
   - *Explanation:* Maintain two heaps to split the window into lower and upper halves. Insert/remove elements in O(log k) time, rebalance heaps to keep sizes equal/differ by 1, and peek the median in O(1). Lazy deletion handles removed elements efficiently.

   **Problem 2: Articulation Points**
   - *Target:* O(V+E) single DFS.
   - *Approach:* Tarjan's algorithm or DFS-based articulation point algorithm using discovery times and low-link values.
   - *Key Conditions:* 
     - Root is AP if it has ≥2 children in DFS tree.
     - Non-root node `u` is AP if it has a child `v` such that `low[v] >= disc[u]` (no back-edge from `v` or its descendants to an ancestor of `u`).
     - Back-edges update `low` values: `low[u] = min(low[u], disc[v])` for back-edges, `low[u] = min(low[u

## Metadata
- **Top-k**: 8
- **Elapsed**: 6164ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
