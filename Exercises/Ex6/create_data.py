# CMPT 353 - Exercise 6
# Arsalan Macknojia

import time
import numpy as np
import pandas as pd
from implementations import all_implementations


def benchmark_sort(random_array):
    benchmark = np.array([])
    for sort in all_implementations:
        st = time.time()
        res = sort(random_array)
        en = time.time()
        runtime = en - st
        benchmark = np.append(benchmark, runtime)
    return benchmark


def main():
    result = pd.DataFrame(columns=['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'])
    for i in range(100):
        random_array = np.random.randint(10000, size=10000)
        result.loc[i] = benchmark_sort(random_array)
    result.to_csv('data.csv', index=False)


if __name__ == '__main__':
    main()

