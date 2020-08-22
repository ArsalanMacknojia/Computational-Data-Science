import re
import sys
from math import sqrt
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), number_of_bytes=int(m.group(2)))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    return log_lines.map(line_to_row).filter(not_none)


def main(in_directory):
    # Get data out of the files into a DataFrame where there is hostname and number of bytes for each request.
    logs = spark.createDataFrame(create_row_rdd(in_directory), schema='hostname:string, number_of_bytes:int')
    logs = logs.cache()

    # Group by hostname; get the number of requests and sum of bytes transferred, to form a data point
    count_requests = logs.groupBy(logs.hostname).agg(functions.count(logs['hostname']).alias('count_requests'))
    sum_request_bytes = logs.groupBy(logs.hostname).agg(functions.sum(logs['number_of_bytes']).alias('sum_request_bytes'))
    data_point = count_requests.join(sum_request_bytes, on='hostname').drop('hostname')

    # Produce six values: 1, x, x^2, y, y^2, xy.
    six_values = data_point.\
        withColumn('ones', functions.lit(1)).\
        withColumn('x^2', data_point['count_requests']**2).\
        withColumn('y^2', data_point['sum_request_bytes']**2).\
        withColumn('xy', data_point['count_requests'] * data_point['sum_request_bytes'])

    # Add these to get the six sums.
    sum_values = six_values.groupBy()
    sum_values = sum_values.agg(
        functions.sum(six_values['ones']),
        functions.sum(six_values['count_requests']),
        functions.sum(six_values['x^2']),
        functions.sum(six_values['sum_request_bytes']),
        functions.sum(six_values['y^2']),
        functions.sum(six_values['xy']))

    # Calculate r
    sum_values = sum_values.first()

    n = sum_values['sum(ones)']
    x = sum_values['sum(count_requests)']
    x_sqr = sum_values['sum(x^2)']
    y = sum_values['sum(sum_request_bytes)']
    y_sqr = sum_values['sum(y^2)']
    xy = sum_values['sum(xy)']

    r = (n*xy - x*y)/(sqrt(n*x_sqr - x**2) * sqrt(n*y_sqr - y**2))
    print("r = %g\nr^2 = %g" % (r, r ** 2))


if __name__ == '__main__':
    in_directory = sys.argv[1]
    main(in_directory)
