# Algorithm Design & Analysis

Design and analyze algorithms for the following problems. For each, provide:
1. A clear description of the approach.
2. Pseudocode or Python implementation.
3. Time and space complexity analysis (Big-O, best/worst/average where relevant).
4. Proof of correctness or counterexample if incorrect.

---

## Problem 1: Merge Intervals with Overlap Count

Given a list of intervals `[start, end]`, produce two outputs:
a) The merged intervals (standard merge intervals problem).
b) For each point on the number line, the maximum number of overlapping intervals at any point.

**Example:**
```
Input:  [[1, 4], [2, 6], [8, 10], [6, 8], [11, 15]]
Merged: [[1, 6], [6, 10], [11, 15]]
Max overlap: 2 (at points 2-4, three intervals [1,4], [2,6], and the boundary)
```

**Constraints:** O(n log n) time, O(n) space.

---

## Problem 2: Sliding Window Median

Given an array of integers and a window size `k`, return the median of each sliding window position.

**Example:**
```
Input:  nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3
Output: [1, -1, -1, 3, 3, 3, 5]
```

**Constraints:** O(n log k) time. Explain your data structure choice.

---

## Problem 3: Graph — Find All Articulation Points

Given an undirected graph, find all articulation points (vertices whose removal disconnects the graph).

**Example:**
```
    0 -- 1 -- 2
         |    |
         3 -- 4
Articulation points: 1 (removing it isolates 0), 4 (removing it isolates 2 from 3)
```

**Constraints:** O(V + E) time using a single DFS. Explain why a naive O(V(V+E)) approach is incorrect or suboptimal.

---

**Scoring criteria:**

| Criterion | Points |
|-----------|--------|
| Correct algorithm | 40% |
| Correct complexity analysis | 25% |
| Working implementation | 20% |
| Proof of correctness / edge case handling | 15% |
