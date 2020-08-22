import re
import string
import sys

from pyspark.sql import SparkSession, functions

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.3'  # make sure we have Spark 2.3+


def main(in_directory, out_directory):
    # Read lines from the files with spark.read.text
    lines = spark.read.text(in_directory)

    # Split the lines into words with the regular expression
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation

    # Create DataFrame to hold words
    words = lines.select(functions.explode(functions.split(lines.value, wordbreak)).alias('words')).collect()
    words_df = spark.createDataFrame(words)

    # Normalize all strings to lower-case and remove empty strings.
    words_df = words_df.select(functions.lower(words_df.words).alias('words')).filter(words_df.words != "")

    # Count the number of times each word occurs.
    words_df = words_df.groupBy('words').agg(functions.count('*').alias('count'))

    # Sort by decreasing count (i.e. frequent words first) and alphabetically if there's a tie.
    words_df = words_df.orderBy(['count', 'words'], ascending=[False, True])

    # Write results as CSV files with the word in the first column, and count in the second
    words_df.write.csv(out_directory)


if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
