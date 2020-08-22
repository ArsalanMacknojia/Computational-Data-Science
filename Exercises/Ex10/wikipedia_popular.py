import os
import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

wiki_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('requests', types.StringType()),
    types.StructField('size', types.LongType()),
])


def get_fn(path):
    head, tail = os.path.split(path)
    return tail[11:-7]


def main(in_directory, out_directory):
    df = spark.read.csv(in_directory, schema=wiki_schema, sep=' ').withColumn('filename', functions.input_file_name())
    path_to_hour = functions.udf(get_fn, returnType=types.StringType())

    df = df.select(df['language'], df['title'], df['requests'], path_to_hour(df['filename']).alias('filename'))
    df = df.filter((df['language'] == 'en') & (df['title'] != 'Main Page') & (
                functions.substring(df['title'], 0, 8) != 'Special:'))

    page_views = df.groupBy('filename', 'title').agg(functions.sum(df['requests']).alias('count'))
    max_views = page_views.groupBy('filename').agg(functions.max(page_views['count']))

    joined = page_views.join(max_views, on='filename')
    most_viewed = joined.filter(joined['count'] == joined['max(count)']).select(joined['filename'], joined['title'],
                                                                                joined['count'])
    most_viewed = most_viewed.cache()

    sort_data = most_viewed.sort('filename', 'title')
    sort_data.write.csv(out_directory, mode='overwrite')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
