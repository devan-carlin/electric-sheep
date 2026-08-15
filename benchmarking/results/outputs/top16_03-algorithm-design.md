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
   - **Topic:** Algorithm Design
   - **Format:** For each problem, provide:
     - Algorithm name/approach
     - Time/Space complexity
     - Brief 1-2 sentence explanation
   - **Constraints:** Focus on approach & complexity only. Brief explanation is fine.
   - **Problems:**
     1. Sliding Window Median (array of n integers, window size k, target O(n log k))
     2. Articulation Points (undirected graph, target O(V+E) single DFS, key conditions back-edge vs tree-edge)
     3. Max Overlapping Intervals (n intervals, target O(n log n), sweep-line technique)

2.  **Problem 1: Sliding Window Median**
   - *Target:* O(n log k)
   - *Approach:* Two heaps (min-heap for upper half, max-heap for lower half) + lazy deletion or balanced BST. Actually, the classic O(n log k) approach uses two heaps with lazy removal, or a balanced BST (like `std::multiset` in C++ or `TreeMap` in Java). The two-heap approach with lazy deletion is O(n log k) time and O(k) space. Alternatively, a balanced BST directly gives O(n log k) time and O(k) space.
   - *Key Data Structure:* Two heaps (min/max) with lazy deletion, or a balanced BST (order statistic tree).
   - *Complexity:* Time: O(n log k), Space: O(k)
   - *Explanation:* Maintain two heaps (max-heap for lower half, min-heap for upper half) to track the median. As the window slides, add/remove elements and rebalance heaps, using lazy deletion to handle removed elements efficiently.

3.  **Problem 2: Articulation Points**
   - *Target:* O(V+E) single DFS
   - *Approach:* DFS-based algorithm using discovery time and low-link values.
   - *Key Conditions:* 
     - For root: articulation point if it has >1 children in DFS tree.
     - For non-root vertex u: articulation point if it has a child v such that `low[v] >= disc[u]` (meaning no back-edge from v or its descendants to an ancestor of u).
    

## Metadata
- **Top-k**: 16
- **Elapsed**: 6466ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
