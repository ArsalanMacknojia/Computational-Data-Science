# CMPT 353 - Exercise 2
# Arsalan Macknojia

import sys
import pandas as pd
import matplotlib.pyplot as plt


def get_csv_as_df(file):
    return pd.read_csv(file, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])


def main(filename1, filename2):
    views_at_1200 = get_csv_as_df(filename1)
    views_at_1300 = get_csv_as_df(filename2)

    # Single plot with two subplots
    plt.figure(figsize=(10, 5))

    # Plot 1: Distribution of Views
    sorted_views_at_1200 = views_at_1200.sort_values(by='views', ascending=False)

    plt.subplot(1, 2, 1)
    plt.plot(sorted_views_at_1200['views'].values)
    plt.title("Popularity Distribution")
    plt.xlabel("Rank")
    plt.ylabel("Views")

    # Plot 2: Daily Views
    views_at_1200['views_at_1300'] = views_at_1300['views']

    plt.subplot(1, 2, 2)
    plt.scatter(views_at_1200['views'].values, views_at_1200['views_at_1300'].values)
    plt.title("Daily Correlation")
    plt.xlabel("Day 1 Views")
    plt.ylabel("Day 2 Views")
    plt.xscale("log")
    plt.yscale("log")

    # Save plot
    plt.savefig('wikipedia.png')


if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    main(filename1, filename2)
