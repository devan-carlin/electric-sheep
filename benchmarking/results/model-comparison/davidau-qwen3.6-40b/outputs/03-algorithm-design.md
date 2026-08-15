# Output: davidau-qwen3.6-40b × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Response
Let me solve each problem step by step.

## 1. Sliding Window Median

**Problem:** Given an array of n integers and a window size k, return the median of each window position. Target: O(n log k).

**Approach:** Use two heaps (min-heap and max-heap) to maintain the elements in the current window.

- Max-heap (left half): stores the smaller half of elements
- Min-heap (right half): stores the larger half of elements

**Algorithm:**
1. Initialize two heaps and balance them so their sizes differ by at most 1
2. For each window position:
   - Add the new element to appropriate heap
   - Remove the element that's sliding out (lazy removal with hash map)
   - Rebalance heaps if needed
   - Compute median from heap tops

**Complexity:**
- Time: O(n log k) - each element is added and removed once, heap operations are O(log k)
- Space: O(k) - storing elements in heaps plus hash map for lazy removal

**Data Structure:** Two heaps (max-heap for left half, min-heap for right half) with a hash map for lazy deletion.

## 2. Articulation Points (Cut Vertices)

**Problem:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS.

**Approach:** Tarjan's algorithm using DFS with discovery time and low values.

**Key Concepts:**
- `disc[u]`: Discovery time of vertex u
- `low[u]`: Lowest discovery time reachable from u (including through back-edges)
- `parent[u]`: Parent of u in DFS tree

**Algorithm:**
1. Run DFS from each unvisited vertex
2. For each vertex u:
   - Set disc[u] = low[u] = timer++
   - For each neighbor v:
     - If v is not visited:
       - parent[v] = u
       - DFS(v)
       - low[u] = min(low[u], low[v])
       - Check articulation point conditions
     - Else if v != parent[u]:
       - low[u] = min(low[u], disc[v])  // Back-edge

**Key Conditions for Articulation Point:**
1. **Root case:** u is root of DFS tree and has 2 or more children in DFS tree

## Metadata
- **Model**: davidau-qwen3.6-40b
- **Elapsed**: 24171ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
- **Gen tok/s**: 21.1
