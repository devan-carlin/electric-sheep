# Output: qwen3.8-27b-16k × 03-algorithm-design

## Prompt
# Algorithm Design — Approach & Complexity Only

For each, give the algorithm name/approach and time/space complexity. Brief 1-2 sentence explanation is fine.

1. **Sliding Window Median:** Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
2. **Articulation Points:** Find all vertices whose removal disconnects an undirected graph. Target: O(V+E) single DFS. What are the key conditions (back-edge vs tree-edge)?
3. **Max Overlapping Intervals:** Given n intervals, find the maximum number of overlapping intervals at any point on the number line. Target: O(n log n). What sweep-line technique?

## Reasoning
We need answer user's request. Need provide algorithm design approach & complexity only for three problems. Need likely concise. Let's reason each.

1 Sliding Window Median: Need O(n log k). Data structure: two heaps (max-heap lower half, min-heap upper half) with lazy deletion, or balanced BST/multiset with order statistics. In C++ multiset + iterator to median? O(n log k) if maintain two multisets lower/upper. Need median each window. Target O(n log k). Data structure? Two heaps with lazy deletion or two balanced BSTs. Explanation: maintain lower half max-heap and upper half min-heap sizes differ by at most 1; median top(s). For sliding, insert new, remove old lazily. Complexity O(n log k), space O(k). Need mention lazy deletion for heaps. If using order-stat tree O(n log k) space O(k). The question asks what data structure? likely two heaps (max/min) with lazy deletion.

2 Articulation Points: Tarjan's algorithm single DFS. Key conditions: root: articulation if has >1 DFS children. Non-root u: articulation if exists tree child v such that low[v] >= disc[u]. Back-edge vs tree-edge: low[v] is min discovery time reachable from v via tree edges and back edges to ancestors (excluding parent tree edge). If low[v] >= disc[u], no back edge from subtree v to ancestor of u, so removing u disconnects subtree. Need mention back-edge to ancestor lowers low; tree-edge to child. Complexity O(V+E), space O(V) recursion/stack.

3 Max Overlapping Intervals: Sweep line with events. For each interval [l,r] create start event +1 at l, end event -1 at r (or r+epsilon depending inclusive/exclusive). Sort events O(n log n), sweep maintaining count, max. If inclusive endpoints, handle starts before ends at same coordinate or use (point, type) with start before end. Complexity O(n log n), space O(n). Could also sort endpoints separately: sort starts and ends, two pointers O(n log n). Sweep-line technique: event sorting with +1/-1. Need answer brief.

Need final likely numbered. Ensure only approach and complexity. Could include data structure. Need maybe mention median definition if even? But not necessary. Let's craft concise but enough.

Double-check complexities:
1: If using two heaps with lazy deletion: each element inserted once and deleted once, heap size O(k + stale) maybe O(n) if not pruning? Actually lazy deletion can accumulate stale entries up to n, but if we prune when top stale, total O(n log n) worst? Target O(n log k). To guarantee O(n log k), use two balanced BSTs/multisets of current window size k, or heaps with lazy deletion and periodic cleanup? Let's analyze. Sliding window median O(n log k) can be done with two heaps and lazy deletion: each element pushed once, popped at most once when stale reaches top. Heap size can be O(n) if many stale not at top? But total operations O(n log n) maybe not O(n log k). However typical accepted O(n log k) with two heaps and lazy deletion? The heap size may be O(n) in worst-case if stale elements remain buried, but log heap size could be log n. But if we delete stale only when top, heap size can grow to n. To guarantee O(n log k), use two balanced BSTs (multisets) containing only active elements, size k, operations O(log k). Or use two heaps with lazy deletion plus maintain sizes and prune; but worst-case heap size? Let's think: each stale element eventually removed? Not necessarily if never top. But total heap size could be O(n). However maybe amortized O(n log k) if we use two heaps and delete arbitrary? Heaps don't support arbitrary delete. Could use indexed priority queues? Or two TreeMaps. The question: "Target: O(n log k). What data structure?" likely expects two heaps (max heap and min heap) with lazy deletion? Many interview answers: maintain two heaps, lazy deletion, O(n log k). But strict complexity? Let's examine.

Sliding window median: Need median of each window. Standard solution: two heaps with lazy deletion. Complexity O(n log k) often stated because each element inserted/deleted O(log k). But lazy deletion: when removing old element, mark in hashmap; when top stale, pop. The heaps can contain stale elements. Number of stale elements in heap at any time could be O(n) if old elements are not at top. But each stale element is popped at most once, total O(n log ?). If heap size O(n), log n. But maybe because we only keep k active plus stale that are in top? No, stale buried can remain. Example: max-heap lower half contains many small stale values? They may be buried under larger active values. Could accumulate. But total operations n, log n not log k. If k small, could be O(n log n). To guarantee O(n log k), use balanced BST/multiset with order statistics, or two heaps plus actual deletion using indexed heap? In languages with multiset, maintain lower and upper multisets of size <=k, remove exact old value O(log k), rebalance O(log k). That is clean. The question asks "What data structure?" Could answer: two balanced BSTs/multisets (or order-statistic trees) partitioning window into lower/upper halves; alternatively two heaps with lazy deletion if acceptable. Since target O(n log k), better say two balanced BSTs/multisets (or two heaps with lazy deletion and careful pruning, but strict O(n log k) via BST). Need maybe mention if using heaps, lazy deletion. The prompt: "Target: O(n log k). What data structure?" It likely expects "two heaps (max-heap and min-heap) with lazy deletion". But if they emphasize target O(n log k), maybe they want "two balanced BSTs / multisets". Let's infer from typical algorithm design questions.

1. Sliding Window Median: Array of n integers, window size k. Return median of each window position. Target: O(n log k). What data structure?
This is LeetCode 480. Common optimal: two heaps with lazy deletion O(n log k). Many say use two heaps. But as noted strict? LeetCode accepted. They often state O(n log k) because heap size O(k)? They ignore stale? Actually with lazy deletion, heap size can be O(n), but maybe they assume delete from heap by marking and when top; still O(n log k) amortized? Let's test worst-case. Suppose k=1. Window median is element. Using two heaps: insert x, remove previous. If we mark previous stale, but if previous not top? For k=1, lower/upper sizes? Let's simulate: maintain lower max heap size 1, upper min heap size 0 maybe. Insert new, rebalance. Remove old mark. If old is top, pop. If not? With k=1, old likely top? Maybe. For larger k, stale can be buried. But total stale in heap maybe O(k)? Let's reason: We only keep active elements partitioned. Stale elements are those removed from window but not popped. Could there be many stale buried? Yes. Example lower max-heap contains active large values and stale small values. Stale small values buried under active large values. As window slides, active large values may leave and become stale, then maybe popped? But stale small values remain. Could accumulate over n. But if k is small, active large values limited, stale small values could accumulate. Example k=2, lower heap size 1? Stale values in lower? Let's construct: values increasing. Window [1,2], lower {1}, upper {2}. Slide remove 1 (stale top lower, pop), insert 3. No accumulation. Decreasing: lower {2}, upper {1}; remove 2 top lower pop. Maybe stale not top if it's not extreme. Suppose lower max-heap size k/2. Stale value in lower not max because active larger values above. It can remain. Over time, active larger values leave, stale may become top and pop. But new stale can accumulate. Could total stale in heap exceed k by O(n)? Possibly. But each stale popped when it reaches top; if never reaches top by end, remains. Heap size O(n). But operations log n. However maybe rebalancing moves elements between heaps, stale can be moved? If stale not top, not moved. So strict O(n log k) not guaranteed with simple lazy heaps. But many sources still say O(n log k) because they assume heap size O(k) or use delete arbitrary.

Alternative: Use two TreeMaps/multisets. In C++ multiset lower/upper, remove exact old value O(log k), rebalance by moving top O(log k). Median is last of lower or first of upper. Complexity O(n log k), space O(k). This is robust. The question "What data structure?" singular? Could be "two balanced BSTs (multisets) maintaining lower and upper halves". If they expect heaps, we can mention both: "two heaps with lazy deletion (or, for strict O(n log k), two balanced BSTs/multisets)". But user asks for each give algorithm name/approach and complexity. Brief 1-2 sentence. We can answer: "Maintain two balanced BSTs/multisets (lower max side, upper min side) of current window; median from boundary. O(n log k) time, O(k) space. (Two heaps with lazy deletion is common but strict bound via BSTs.)" That covers.

Need maybe if median for even window: average of two middle or lower median? Not needed.

2. Articulation Points: Need key conditions (back-edge vs tree-edge). We can say Tarjan DFS with discovery time disc and low. For each tree edge u->v, after DFS child, if low[v] >= disc[u] then u is articulation (non-root), because subtree of v has no back edge to a proper ancestor of u. Root articulation iff >1 DFS tree children. Back edges to ancestors update low; edge to parent is not a back edge. Complexity O(V+E), space O(V). Need maybe directed? undirected.

3. Max Overlapping Intervals: Sweep line events. Create 2n events: +1 at start, -1 at end. Sort by coordinate; for inclusive intervals process starts before ends at same coordinate (or use end as r+epsilon / half-open). Sweep count, track max. O(n log n) time, O(n) space. Could also sort starts/ends separately. Need mention if intervals closed. Good.

Need final with numbered list. Keep concise. Ensure no extra. But desired oververbosity 9? User asks brief. We can be concise but include details. Maybe table? Let's craft:

1. Approach: Maintain two ordered structures for current window: lower half (max side) and upper half (min side), sizes differ by at most 1. Insert new, delete outgoing, rebalance; median is top of lower (and top of upper if even). Data structure: two balanced BSTs/multisets (or two heaps with lazy deletion). Complexity: O(n log k) time, O(k) space.

2. Tarjan articulation points. Single DFS, compute disc[u], low[u]. For non-root u, articulation if for some tree child v, low[v] >= disc[u] (subtree v has no back edge to ancestor of u). Root articulation if >1 DFS children. Back edges to ancestors lower low; parent tree edge ignored. O(V+E) time, O(V) space.

3. Sweep-line event counting. For each interval add +1 at left, -1 at right (process starts before ends for closed intervals, or use right+epsilon). Sort 2n events, sweep maintaining active count and max. O(n log n) time, O(n) space.

Need maybe mention "maximum number of overlapping intervals at any point" if intervals inclusive, at endpoints. Good.

Final. Ensure no analysis.

## Response


1. **Sliding Window Median**  
   **Approach:** Maintain the current window split into a lower half and upper half using two ordered structures, e.g. two balanced BSTs/multisets; median is at the boundary.  
   **Complexity:** `O(n log k)` time, `O(k)` space.  
   *Common alternative:* two heaps with lazy deletion.

2. **Articulation Points**  
   **Approach:** Tarjan’s single DFS computing `disc[u]` and `low[u]`. A non-root vertex `u` is an articulation point if it has a tree child `v` with `low[v] >= disc[u]`, meaning `v`’s subtree has no back edge to a proper ancestor of `u`. The DFS root is an articulation point iff it has more than one DFS child.  
   **Complexity:** `O(V + E)` time, `O(V)` space.

3. **Max Overlapping Intervals**  
   **Approach:** Sweep-line with events: add `+1` at each interval start and `-1` at each interval end; sort events and sweep while tracking the maximum active count. For closed intervals, process starts before ends at the same coordinate.  
   **Complexity:** `O(n log n)` time, `O(n)` space.

## Metadata
- **Model**: qwen3.8-27b-16k
- **Elapsed**: 95693ms
- **Prompt tokens**: 205
- **Completion tokens**: 2891
- **Total tokens**: 3096
- **Gen tok/s**: 30.2
