"""Q1: 'People You Might Know' friend recommendation with Spark.

Pipeline: each adjacency line is expanded into two kinds of key-value pairs
keyed by an unordered user pair -- a marker (-1) for every existing friendship
and a +1 for every pair of users that share the line's owner as a mutual
friend.  A reduceByKey that propagates the -1 marker drops pairs that are
already friends and sums the mutual-friend counts for the rest.  Each surviving
pair is emitted in both directions, grouped by user, and the top 10 candidates
per user are taken in decreasing count / increasing id order.
"""

import os
import shutil
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
from pyspark import SparkContext

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "soc-LiveJournal1Adj.txt")
OUT = os.path.join(HERE, "output")

N_RECOMMENDATIONS = 10
ALREADY_FRIENDS = -1
REPORT_USERS = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]


def parse_line(line):
    parts = line.split("\t")
    user = int(parts[0])
    friends = [int(f) for f in parts[1].split(",") if f] if len(parts) > 1 else []
    return user, friends


def emit_pairs(record):
    """Yield (pair, value) for direct friendships and mutual-friend evidence."""
    user, friends = record
    for friend in friends:
        yield (min(user, friend), max(user, friend)), ALREADY_FRIENDS
    for i in range(len(friends)):
        for j in range(i + 1, len(friends)):
            a, b = friends[i], friends[j]
            yield (min(a, b), max(a, b)), 1


def combine(a, b):
    if a == ALREADY_FRIENDS or b == ALREADY_FRIENDS:
        return ALREADY_FRIENDS
    return a + b


def top_recommendations(candidates):
    ranked = sorted(candidates, key=lambda t: (-t[1], t[0]))
    return [str(user) for user, _ in ranked[:N_RECOMMENDATIONS]]


def main():
    sc = SparkContext("local[*]", "FriendsRecommendation")
    sc.setLogLevel("WARN")

    adjacency = sc.textFile(SRC).map(parse_line).cache()

    recommendations = (adjacency
                       .flatMap(emit_pairs)
                       .reduceByKey(combine)
                       .filter(lambda kv: kv[1] > 0)
                       .flatMap(lambda kv: [(kv[0][0], (kv[0][1], kv[1])),
                                            (kv[0][1], (kv[0][0], kv[1]))])
                       .groupByKey()
                       .mapValues(top_recommendations))

    # Users with no second-degree friends still need an (empty) output line.
    result = (adjacency
              .map(lambda kv: (kv[0], []))
              .leftOuterJoin(recommendations)
              .mapValues(lambda v: v[1] if v[1] is not None else [])
              .sortByKey())

    shutil.rmtree(OUT, ignore_errors=True)
    (result
     .map(lambda kv: "%d\t%s" % (kv[0], ",".join(kv[1])))
     .saveAsTextFile(OUT))

    wanted = set(REPORT_USERS)
    answers = dict(result.filter(lambda kv: kv[0] in wanted).collect())
    for user in REPORT_USERS:
        print("%d\t%s" % (user, ",".join(answers.get(user, []))))

    sc.stop()


if __name__ == "__main__":
    main()
