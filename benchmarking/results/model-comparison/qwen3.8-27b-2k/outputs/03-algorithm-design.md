# Output: qwen3.8-27b-2k × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Reasoning
We need answer user's request: For each, give algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation. Need likely concise. Need ensure correctness.

1 Sliding Window Median target O(n log k). Data structure? Need maintain two heaps with lazy deletion or balanced BST/multiset. In Python maybe two heaps with lazy deletion; in C++ multiset. Complexity O(n log k), space O(k). Need mention median of each window. Approach: sliding window with two balanced structures: max-heap lower half, min-heap upper half, rebalance sizes, lazy deletions for outgoing elements. Or order-statistic tree. Data structure: two heaps with lazy deletion or balanced BST.

2 Articulation Points: Tarjan's algorithm DFS. Key conditions: root: articulation if >1 DFS children. Non-root: articulation if exists child v such that low[v] >= disc[u] (no back edge from subtree of v to strict ancestor of u). Need mention back-edge vs tree-edge: low[v] is min discovery time reachable from subtree via tree edges and back edges; if low[v] >= disc[u], removing u disconnects v's subtree. For root condition children count >1. Complexity O(V+E), space O(V) recursion/stack.

3 Max Overlapping Intervals: sweep line with events. Create 2n events: start +1, end -1. Sort by coordinate; for same coordinate process starts before ends if intervals closed? Need maximum overlapping at any point. If intervals [l,r] inclusive, process start before end at same coordinate. If half-open, end before start. Usually maximum number of overlapping intervals: sort endpoints, +1 at start, -1 at end, tie start before end for closed intervals. Complexity O(n log n), space O(n). Could also difference array if coordinate compressed. Need answer.

Need final likely numbered. Include algorithm name/approach and time/space. Brief. Ensure target complexities. Maybe mention for 1 data structure explicitly. Let's craft with enough detail but concise.

## Response


1. **Sliding Window Median**  
   **Approach:** Maintain the window in two balanced structures: a max-heap for the lower half and a min-heap for the upper half, with lazy deletion for elements leaving the window.  
   **Complexity:** `O(n log k)` time, `O(k)` space.

2. **Articulation Points**  
   **Approach:** Use Tarjan’s DFS with discovery times and low-link values. A non-root vertex `u` is an articulation point if it has a child `v` with `low[v] >= disc[u]`; the root is an articulation point if it has more than one DFS child.  
   **Complexity:** `O(V + E)` time, `O(V)` space.

3. **Max Overlapping Intervals**  
   **Approach:** Use a sweep line over interval endpoints: add `+1` at each start and `-1` at each end, sorting events and processing starts before ends for closed intervals.  
   **Complexity:** `O(n log n)` time, `O(n)` space.

## Metadata
- **Model**: qwen3.8-27b-2k
- **Elapsed**: 21967ms
- **Prompt tokens**: 205
- **Completion tokens**: 667
- **Total tokens**: 872
- **Gen tok/s**: 30.3
