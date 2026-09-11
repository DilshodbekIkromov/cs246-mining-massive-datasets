import os, re, shutil, sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
from pyspark import SparkContext

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "pg100.txt")
out = os.path.join(here, "output")
shutil.rmtree(out, ignore_errors=True)

sc = SparkContext("local[*]", "LetterCount")
sc.setLogLevel("WARN")
# every line here returns new RDD 
counts = (sc.textFile(src) 
          .flatMap(lambda l: re.split(r"[^a-zA-Z]+", l))
          .filter(lambda w: w)
          .map(lambda w: (w[0].lower(), 1)) #turns each word into a pair (key, value)
          .reduceByKey(lambda a, b: a + b) # takes 2 values for the same key and combines them 
          .sortByKey().cache())
counts.saveAsTextFile(out) # first action which trigger excecution
for letter, n in counts.collect():
    print(letter, n)
sc.stop()
