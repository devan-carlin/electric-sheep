# Output: top-64 × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Response
Here's a thinking process:

1.  **Analyze the Request:**
    *   The user wants algorithm design for three specific problems.
    *   Focus: "Approach & Complexity Only"
    *   Format: Algorithm name/approach, time/space complexity, brief 1-2 sentence explanation.
    *   Constraints: Target complexities specified for each problem.
    *   Questions: "What data structure?" / "What are the key conditions?" / "What sweep-line technique?"

2.  **Problem 1: Sliding Window Median**
    *   Input: Array of n integers, window size k.
    *   Output: Median of each window position.
    *   Target Complexity: O(n log k)
    *   Key Question: What data structure?
    *   *Thinking:* Median requires sorting or finding the middle element. For a sliding window, we need to maintain a set of k elements efficiently.
    *   *Approach:* Two heaps (or balanced BSTs) approach. One heap for lower half, one heap for upper half. Or a single balanced BST with median tracking.
    *   *Complexity:* O(n log k) is achievable if we use a balanced BST (like AVL, Red-Black, or a specialized median-tracking structure) or two heaps. Insert/remove takes O(log k). We slide n times. Total O(n log k).
    *   *Data Structure:* Balanced Binary Search Tree (BST) with median tracking, or two heaps (min/max heaps).
    *   *Explanation:* Maintain a balanced BST (or two heaps) tracking the median element; insert/remove each window shift takes O(log k), yielding O(n log k) total.

3.  **Problem 2: Articulation Points**
    *   Input: Undirected graph.
    *   Output: Vertices whose removal disconnects the graph.
    *   Target Complexity: O(V+E) single DFS.
    *   Key Question: What are the key conditions (back-edge vs tree-edge)?
    *   *Thinking:* Standard algorithm is DFS with discovery/finish times. Articulation point condition: A vertex is an articulation point if it has a child in DFS tree whose finish time is >= discovery time of parent (no back-edge from child to parent or ancestors). Or, root is articulation if has >1 children.
    *   *Approach

## Metadata
- **Top-k**: 64
- **Elapsed**: 8282ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
