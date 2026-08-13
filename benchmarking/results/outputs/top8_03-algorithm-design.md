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
   - **Format:** Algorithm Design — Approach & Complexity Only
   - **Requirement:** For each problem, give algorithm name/approach, time/space complexity, and a brief 1-2 sentence explanation.
   - **Problem 1:** Sliding Window Median, n integers, window k, target O(n log k). What data structure?
   - **Problem 2:** Articulation Points, undirected graph, target O(V+E) single DFS. What are key conditions?
   - **Problem 3:** Max Overlapping Intervals, n intervals, target O(n log n). What sweep-line technique?
   - **Constraint:** Only approach & complexity, brief explanation fine.

2.  **Deconstruct Each Problem & Formulate Answer:**

   *Problem 1: Sliding Window Median*
   - Target: O(n log k)
   - Data structure: Two heaps (max-heap for lower half, min-heap for upper half) or balanced BST (like `std::multiset` in C++ or `SortedList` in Python via `bisect`/`sortedcontainers`). Two heaps is the classic O(n log k) approach.
   - Approach: Maintain two heaps balancing sizes ≤1. Median is top of heaps. Slide by removing outgoing element (lazy removal or rebalance) and inserting new element.
   - Complexity: Time O(n log k), Space O(k).
   - Explanation: Keep lower/upper halves balanced; median is O(1) peek, insert/delete O(log k). Lazy removal handles sliding window deletions without extra log factor.

   *Problem 2: Articulation Points*
   - Target: O(V+E) single DFS
   - Key conditions: Uses discovery time `disc[u]` and low-link `low[u]`. A tree edge (u,v) is a bridge/AP condition if `low[v] >= disc[u]`. Root is AP if it has ≥2 DFS children. Back-edges update `low` via `low[u] = min(low[u], disc[v])`.
   - Complexity: Time O(V+E), Space O(V) for recursion stack/arrays.
   - Explanation: DFS tracks earliest reachable ancestor; if a subtree can't reach above u, u is critical. Back-edges lower low-values, preventing false positives.

   *Problem 3:

## Metadata
- **Top-k**: 8
- **Elapsed**: 6092ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
