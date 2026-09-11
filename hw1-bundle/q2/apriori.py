"""Q2 (d,e): A-priori on the online browsing sessions.

Support threshold is 100.  Pass 1 counts items, pass 2 counts pairs built from
frequent items, pass 3 counts triples built from frequent pairs.  From the
frequent pairs and triples we derive the association rules and their
confidence scores.
"""

import os
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "browsing.txt")

SUPPORT = 100
TOP_N = 5


def load_baskets(filename):
    baskets = []
    with open(filename) as f:
        for line in f:
            items = set(line.split())
            if items:
                baskets.append(items)
    return baskets


def frequent_items(baskets, support):
    counts = Counter()
    for basket in baskets:
        counts.update(basket)
    return {item: c for item, c in counts.items() if c >= support}


def frequent_sets(baskets, candidates_from_basket, support):
    """Count candidate itemsets over the baskets and keep the frequent ones."""
    counts = Counter()
    for basket in baskets:
        counts.update(candidates_from_basket(basket))
    return {itemset: c for itemset, c in counts.items() if c >= support}


def pair_candidates(l1):
    def gen(basket):
        items = sorted(item for item in basket if item in l1)
        return combinations(items, 2)
    return gen


def triple_candidates(l2):
    def gen(basket):
        items = sorted(item for item in basket if item in l1_items)
        for triple in combinations(items, 3):
            # Monotonicity: every 2-subset of a frequent triple is frequent.
            if all(pair in l2 for pair in combinations(triple, 2)):
                yield triple
    l1_items = {item for pair in l2 for item in pair}
    return gen


def pair_rules(l1, l2):
    """conf(X => Y) = support({X, Y}) / support(X) for both directions."""
    rules = []
    for (x, y), support_xy in l2.items():
        rules.append(((x,), y, support_xy / l1[x]))
        rules.append(((y,), x, support_xy / l1[y]))
    return rules


def triple_rules(l2, l3):
    """conf((X, Y) => Z) = support({X, Y, Z}) / support({X, Y})."""
    rules = []
    for triple, support_xyz in l3.items():
        for held_out in triple:
            lhs = tuple(item for item in triple if item != held_out)
            rules.append((lhs, held_out, support_xyz / l2[lhs]))
    return rules


def top_rules(rules, n):
    # Decreasing confidence, ties broken lexicographically on the left side
    # (first item, then second) and finally on the right side.
    return sorted(rules, key=lambda r: (-r[2], r[0], r[1]))[:n]


def show(title, rules):
    print(title)
    for lhs, rhs, conf in rules:
        print("  (%s) => %s\tconf = %.6f" % (", ".join(lhs), rhs, conf))
    print()


def main():
    baskets = load_baskets(SRC)
    print("baskets: %d" % len(baskets))

    l1 = frequent_items(baskets, SUPPORT)
    print("|L1| = %d" % len(l1))

    l2 = frequent_sets(baskets, pair_candidates(l1), SUPPORT)
    print("|L2| = %d" % len(l2))

    l3 = frequent_sets(baskets, triple_candidates(l2), SUPPORT)
    print("|L3| = %d\n" % len(l3))

    show("Top %d pair rules [2(d)]:" % TOP_N, top_rules(pair_rules(l1, l2), TOP_N))
    show("Top %d triple rules [2(e)]:" % TOP_N, top_rules(triple_rules(l2, l3), TOP_N))


if __name__ == "__main__":
    main()
