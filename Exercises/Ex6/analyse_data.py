import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():
    data = pd.read_csv('data.csv')

    reshape_data = pd.melt(data)
    posthoc = pairwise_tukeyhsd(reshape_data['value'], reshape_data['variable'])
    print(posthoc)
    posthoc.plot_simultaneous()
    plt.show()


if __name__ == '__main__':
    main()
