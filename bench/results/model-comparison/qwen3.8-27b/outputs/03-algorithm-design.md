# Output: qwen3.8-27b × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Reasoning
We need answer user's request: For each, give algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation. Need likely concise. Need ensure correctness.

1 Sliding Window Median: target O(n log k). Data structure? Need maintain two heaps (max-heap lower half, min-heap upper half) with lazy deletion or balanced BST/multiset. Complexity O(n log k), space O(k). Explanation: keep lower/upper halves balanced, median from top(s), remove expired via lazy deletion.

2 Articulation Points: Tarjan's algorithm single DFS. Key conditions: root with >1 DFS children; non-root vertex u where for child v, low[v] >= disc[u]. Back-edge vs tree-edge: low[v] is min discovery time reachable from subtree via back edges; if no back edge from subtree to ancestor of u, low[v] >= disc[u]. Need mention ignore parent tree edge. Complexity O(V+E), space O(V) recursion/stack.

3 Max Overlapping Intervals: sweep line with events: start +1, end -1; sort events O(n log n), scan max prefix. Need tie handling: if end and start same point, depending on closed intervals? Usually intervals [start,end], max overlap at point; if end at x and start at x, they overlap at x if closed. To count correctly, process starts before ends at same coordinate (or use end exclusive). If half-open, process ends before starts. Need mention. Complexity O(n log n), space O(n) for events (or O(1) extra if sort in place? events O(n)).

Need final answer likely numbered. Keep brief. Ensure target complexities. Maybe include data structure names.

Need think if any nuance: Sliding window median O(n log k) with two heaps and lazy deletion. If using balanced BST with order statistics maybe O(n log k). Data structure: two heaps (max/min) or two balanced BSTs. The question asks What data structure? Answer: two heaps with lazy deletion (or two multisets). Need mention median of each window.

Articulation points: key conditions: root: >1 children in DFS tree; non-root: exists child v such that low[v] >= disc[u]. Back-edge: edge to ancestor (not parent) updates low; tree-edge to child: after DFS, if low[child] >= disc[u] then u articulation. Need maybe low[v] < disc

## Response


## Metadata
- **Model**: qwen3.8-27b
- **Elapsed**: 16794ms
- **Prompt tokens**: 205
- **Completion tokens**: 512
- **Total tokens**: 717
- **Gen tok/s**: 30.4
