import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

comments_schema = types.StructType([
    types.StructField('author', types.StringType()),
    types.StructField('score', types.LongType()),
    types.StructField('subreddit', types.StringType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    # Calculate the average score for each subreddit
    grouped_data = comments.groupBy(comments['subreddit'])
    averages = grouped_data.agg(functions.avg(comments['score']))

    # Exclude any subreddits with average score ≤0
    averages = averages.filter(averages['avg(score)'] > 0)
    averages = functions.broadcast(averages)

    # Join the average score to the collection of all comments.
    # Divide score with average score to get the relative score.
    joined = comments.join(averages, on='subreddit')
    rel_score = joined.withColumn("rel_score", joined['score'] / joined['avg(score)'])
    rel_score = rel_score.cache()

    # Determine the max relative score for each subreddit.
    grouped = rel_score.groupBy(rel_score['subreddit'])
    max_rel_score = grouped.agg(functions.max(rel_score['rel_score']).alias('rel_score'))
    max_rel_score = functions.broadcast(max_rel_score)

    # Join again to get the best comment on each subreddit to get the author.
    best_author = rel_score.join(max_rel_score, on='rel_score').drop(max_rel_score.subreddit)
    best_author = best_author.select(best_author['subreddit'], best_author['author'], best_author['rel_score'])
    best_author.write.json(out_directory, mode='overwrite')


if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
