# Output: top-16 × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Response
Here's a thinking process:

1.  **Analyze User Request:**
   - Format: Algorithm name/approach + time/space complexity. Brief 1-2 sentence explanation.
   - Three specific problems:
     1. Sliding Window Median: O(n log k) target. What data structure?
     2. Articulation Points: O(V+E) single DFS. Key conditions (back-edge vs tree-edge)?
     3. Max Overlapping Intervals: O(n log n). What sweep-line technique?
   - Constraint: "Approach & Complexity Only" - keep it concise.

2.  **Problem 1: Sliding Window Median**
   - Target: O(n log k)
   - Known approach: Two heaps (max-heap for lower half, min-heap for upper half) or balanced BST (like `std::multiset` in C++ or `TreeMap` in Java). Two heaps is the classic O(n log k) approach.
   - Complexity: Time O(n log k), Space O(k).
   - Explanation: Maintain two heaps partitioning the window into lower/upper halves; rebalance after sliding by removing the outgoing element and adding the new one, keeping sizes within 1 of each other.
   - Check: Matches target. Two heaps is standard. BST also works but heaps are more common in interviews. I'll specify Two Heaps.

3.  **Problem 2: Articulation Points**
   - Target: O(V+E) single DFS
   - Known approach: DFS with discovery time and low-link values.
   - Key conditions: A vertex u is an articulation point if (1) it's the DFS root and has >=2 children, or (2) it's not root and has a child v where `low[v] >= disc[u]` (no back-edge from v's subtree to an ancestor of u).
   - Complexity: Time O(V+E), Space O(V) for recursion stack/arrays.
   - Explanation: Track discovery time and lowest reachable ancestor via back-edges; a tree-edge to child v is critical if `low[v] >= disc[u]`, meaning v's subtree cannot escape u's ancestor.
   - Check: Matches target. Conditions correctly distinguished.

4.  **Problem 3: Max Overlapping Intervals**
   - Target: O(n log n)
   - Known approach:

## Metadata
- **Top-k**: 16
- **Elapsed**: 6559ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
