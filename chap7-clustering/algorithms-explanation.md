These five are the standard progression from *Mining of Massive Datasets* (Ch. 7). The way to keep them straight is to notice what each one is fighting against:

| Algorithm | Space | Data fits in RAM? | Cluster shape assumed | Cluster stored as |
|---|---|---|---|---|
| k-means | Euclidean | yes | roughly round | centroid |
| BFR | Euclidean | no (disk) | axis-aligned Gaussian | (N, SUM, SUMSQ) |
| CURE | Euclidean | no (disk) | any shape | set of representative points |
| GRGPF | non-Euclidean | no (disk) | any | clustroid + ROWSUMs, in a tree |
| BDMO | either | stream | inherited from base algo | buckets of cluster records |

---

## 1. k-means

**Intuition:** put k flags on the map, everyone walks to the nearest flag, then each flag moves to the middle of its crowd. Repeat until nobody moves.

**Steps**
1. Pick k initial centroids.
2. Assign every point to the closest centroid.
3. Recompute each centroid as the mean of the points assigned to it.
4. Repeat 2–3 until assignments stop changing.

**Choosing the initial k points** (matters a lot, k-means is greedy and never revisits):
- Pick points that are as far from each other as possible: pick one at random, then repeatedly pick the point whose distance to the nearest already-chosen point is maximum.
- Or: sample the data, cluster the sample hierarchically, take the centroids of those clusters.

**Choosing k:** run for several k and measure average cluster diameter (or radius). As k grows toward the true number, the measure drops fast; past the true number it flattens out. To avoid trying every value, double k (1, 2, 4, 8, …) until the measure stops improving, then binary-search between the last two. That gives you the right k in about 2·log(k) runs.

**Weakness:** assumes convex, similar-sized, roughly spherical clusters. An S-shaped cluster or a ring around a blob will be split wrongly.

---

## 2. BFR (Bradley–Fayyad–Reina)

**Intuition:** k-means for data far bigger than RAM. Instead of keeping points, keep enough statistics about each cluster that you can reconstruct its centroid and spread, then throw the points away.

**Assumption:** each cluster is a normal distribution centered on its centroid, and it is *axis-aligned*. The standard deviation may differ per dimension, but the cluster cannot lie at a diagonal. This is what makes the (N, SUM, SUMSQ) summary sufficient.

**The three sets it maintains in memory:**
- **DS (Discard Set):** points already absorbed into one of the k clusters. Points deleted, only the summary kept.
- **CS (Compression Set):** "miniclusters" of points that are close to each other but not close to any of the k centroids. Summarized, but not yet committed to any cluster.
- **RS (Retained Set):** lonely points that belong to nothing yet. Kept as actual points.

**The summary (2d + 1 numbers per cluster, d = dimensions):**
- `N` = number of points
- `SUM[i]` = sum of the i-th coordinate over all points
- `SUMSQ[i]` = sum of squares of the i-th coordinate

From these: centroid`[i] = SUM[i]/N`, and variance`[i] = SUMSQ[i]/N − (SUM[i]/N)²`. This form is chosen because merging two clusters is just componentwise addition, and adding one point is a constant-time update.

**Steps (per chunk of data read from disk):**
1. Load one memory-sized chunk.
2. For every point close enough to one of the k centroids, add it to that cluster's DS summary (`N++`, `SUM += p`, `SUMSQ += p²`) and discard the point.
3. Cluster the leftovers *together with the current RS* using any in-memory method. Groups of 2+ points become new CS miniclusters; singletons go back into RS.
4. Try to merge the new miniclusters into existing CS miniclusters (merge if the combined variance stays under a threshold).
5. On the last chunk only: fold every CS minicluster and RS point into the nearest DS cluster, or declare them outliers.

**"Close enough" is measured with Mahalanobis distance,** not Euclidean:

$$d(p,c) = \sqrt{\sum_{i=1}^{d}\left(\frac{p_i - c_i}{\sigma_i}\right)^2}$$

You normalize each dimension by that cluster's own spread in that dimension. A cluster that is wide in x and narrow in y accepts distant-in-x points and rejects the same distance in y. Accept the point if this distance is below a threshold like 3 or 4. At 4, for normally distributed data the odds of a genuine member being rejected are under one in a million.

---

## 3. CURE (Clustering Using REpresentatives)

**Intuition:** a single centroid cannot describe a banana or a ring. So describe a cluster by a handful of scattered "ambassador" points spread around its boundary, pulled slightly inward.

Still Euclidean, but no assumption about shape or axis alignment.

**Phase 1 — initialize (in memory, on a sample):**
1. Take a random sample small enough to fit in RAM.
2. Cluster it hierarchically: repeatedly merge the two closest clusters. Stop when you have your target clusters.
3. For each cluster, pick a small set of **representative points** (typically ~4), chosen to be as far from one another as possible: take the point farthest from the centroid, then the point farthest from the ones chosen so far, and so on.
4. **Shrink** each representative by moving it a fixed fraction (usually 20%) of the way toward the cluster's centroid. This is the trick that makes CURE robust: an outlier picked as a representative gets dragged back inside, and the representatives end up just inside the cluster's real boundary.

**Phase 2 — merge:**
5. Merge any two clusters that have a pair of representative points (one from each) closer than some threshold. Repeat until nothing merges.

This is why CURE separates a ring from the blob inside it. The ring's representatives are all on the ring, and the blob's are all in the blob, so no cross pair is close. But two halves of the same long snake-shaped cluster will have adjacent representatives, so they merge correctly.

**Phase 3 — assign:**
6. Stream every point on disk once. Assign each point to the cluster containing the **closest representative point** (not the closest centroid). One pass, no re-reading.

---

## 4. GRGPF (Ganti, Ramakrishnan, Gehrke, Powell, French)

**Intuition:** what if there is no centroid, because you have no coordinates, only a distance function (edit distance on strings, Jaccard on sets)? You can't average points. So elect an actual data point as the cluster's representative, and build an index over clusters so you don't compare a new point against all of them.

**Clustroid instead of centroid.** Define

$$\text{ROWSUM}(p) = \sum_{q \in C} d(p,q)^2$$

The **clustroid** is the member point with the smallest ROWSUM. It's the point that's most "central" in the only sense available.

**What is stored per cluster:**
- `N` = number of points
- the clustroid and its ROWSUM
- the **k closest** points to the clustroid, each with its ROWSUM (if the cluster shifts, the new clustroid is probably one of these)
- the **k furthest** points from the clustroid, each with its ROWSUM (used when merging: the merged clustroid tends to lie toward the other cluster, so it's likely one of these outer points)

Cluster radius = `sqrt(ROWSUM(clustroid)/N)`.

**The tree.** Cluster representations are stored in a B-tree/R-tree-like structure. Leaves hold as many cluster representations as fit in a block. Interior nodes hold a *sample* of the clustroids of the clusters below them, plus child pointers. Nearby clusters are kept in nearby leaves.

**Steps:**
1. **Initialize:** take a main-memory sample, cluster it hierarchically, take clusters of a reasonable size, compute their representations, and load them into the tree.
2. **Insert a point p:** descend from the root, at each node following the child whose sampled clustroids are closest to p. At the leaf, pick the cluster with the nearest clustroid.
3. **Update that cluster:** `N++`, and for each of the 2k+1 stored points q, do `ROWSUM(q) += d(p,q)²`.
4. **Estimate p's own ROWSUM** without touching the discarded points, using the curse-of-dimensionality assumption that in high dimensions any two vectors from the clustroid c are roughly perpendicular, so `d(p,q)² ≈ d(p,c)² + d(c,q)²`. Summing over all q:

$$\text{ROWSUM}(p) = \text{ROWSUM}(c) + N \cdot d(p,c)^2$$

   If this beats one of the stored closest/furthest points, p replaces it. If it beats the clustroid, p becomes the new clustroid.
5. **Split** when a cluster's radius exceeds the limit: break it in two. That may overflow the leaf, which splits, which may cascade up the tree exactly like a B-tree.
6. **Merge when memory runs out:** raise the radius limit and merge clusters (preferring siblings in the tree). To find the merged clustroid without the raw points, evaluate the 2k candidate outer points using the same right-angle assumption:

$$\text{ROWSUM}_{C_1 \cup C_2}(p) = \text{ROWSUM}_{C_1}(p) + N_2\left(d(p,c_1)^2 + d(c_1,c_2)^2\right) + \text{ROWSUM}_{C_2}(c_2)$$

   Pick the candidate with the smallest value as the new clustroid, then rebuild the closest/furthest lists from the available candidates.

---

## 5. BDMO (Babcock, Datar, Motwani, O'Callaghan)

**Intuition:** clustering a stream, where you're asked "cluster the last m arrivals" at any moment. It's the DGIM bucket trick, except each bucket stores *clusters* instead of a count of 1s. Recent data is kept at fine granularity, old data is progressively coarsened.

**Bucket structure:**
- Sizes are `p, 2p, 4p, 8p, …` where p is a power of 2 (in plain DGIM, p = 1).
- There are **one or two buckets of each size**, sizes strictly increasing as you go back in time. This gives O(log N) buckets.
- Each bucket stores: its size, the timestamp of its most recent point, and a set of **cluster records** for the points it covers.
- Each cluster record stores: number of points, centroid or clustroid, plus whatever is needed to merge and to estimate radius (for example sum of distances or sum of squares to the centroid).

**Steps:**
1. **Create:** every p arrivals, form a new bucket from the newest p points. Cluster those p points with any in-memory algorithm (k-means, hierarchical) and store the resulting cluster records.
2. **Expire:** drop any bucket whose timestamp is more than N time units old.
3. **Merge:** if three buckets of size p now exist, merge the two oldest into a bucket of size 2p. If that creates three of size 2p, merge again, and so on up the chain. Each new arrival triggers at most O(log N) merges.
4. **Merge the cluster records** inside merged buckets: pair up clusters from the two buckets whose centroids are closest, and combine them. The merged centroid is the weighted average by point count. Refuse a merge if the resulting cluster would be too large (radius or diameter above threshold), so unrelated clusters don't get glued together.
5. **Answer a query for the last m points:** you can't isolate exactly m points, so take the smallest set of buckets that fully covers the last m. The oldest bucket in that set spills over into older data, but since bucket sizes at most double, you cover at most 2m points. Pool all cluster records from those buckets, merge the ones with nearby centroids, and return the result.

**The accuracy assumption:** the clusters don't drift much over time. That's what makes the extra pre-window points in the oldest bucket harmless.

---

**One-line takeaways**

- **k-means:** move flags to the middle of their crowd, repeat.
- **BFR:** k-means where you keep statistics, not points, plus a holding pen (CS/RS) for the undecided.
- **CURE:** describe a cluster by scattered representatives pulled 20% inward, so odd shapes survive.
- **GRGPF:** no coordinates, so elect a real point (clustroid) and index clusters in a tree.
- **BDMO:** DGIM buckets, but each bucket carries cluster summaries instead of a count.