# Output: davidau-qwen3.6-27b × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Response
Here's a thinking process that leads to the suggested response:

1.  **Analyze the Request:**
    *   **Topic:** Algorithm Design.
    *   **Format:** Approach/Name, Time/Space Complexity, Brief Explanation (1-2 sentences).
    *   **Constraint:** Focus on approach and complexity only.
    *   **Problems:**
        1.  Sliding Window Median ($O(n \log k)$).
        2.  Articulation Points ($O(V+E)$ DFS).
        3.  Max Overlapping Intervals ($O(n \log n)$ Sweep-line).

2.  **Problem 1: Sliding Window Median**
    *   *Goal:* Maintain median of a moving window of size $k$ in an array of size $n$.
    *   *Target Complexity:* $O(n \log k)$.
    *   *Key Challenge:* Insertions and deletions happen at both ends (or rather, one end enters, one leaves), and we need the middle element efficiently.
    *   *Data Structure:* We need something that keeps elements sorted or allows finding the median quickly.
        *   Option A: Balanced BST (e.g., `std::multiset` in C++, `TreeSet` in Java). Insert/Delete is $O(\log k)$, finding median is $O(1)$ or $O(\log k)$ depending on implementation (iterator arithmetic). Total: $O(n \log k)$.
        *   Option B: Two Heaps (Max-Heap for lower half, Min-Heap for upper half). Insert/Delete is $O(\log k)$. Balancing is $O(\log k)$. Finding median is $O(1)$. Total: $O(n \log k)$.
    *   *Selection:* Two Heaps is the classic interview answer for this specific constraint, though a Balanced BST is also valid. I will mention Two Heaps as it's very distinct for "median" problems.
    *   *Drafting the response:*
        *   Approach: Two Heaps (Max-Heap for left half, Min-Heap for right half).
        *   Complexity: Time $O(n \log k)$, Space $O(k)$.
        *   Explanation: Maintain two heaps of size $\approx k/2$ to track the middle elements; slide the window by removing the outgoing

## Metadata
- **Model**: davidau-qwen3.6-27b
- **Elapsed**: 16223ms
- **Prompt tokens**: 163
- **Completion tokens**: 512
- **Total tokens**: 675
- **Gen tok/s**: 31.5
