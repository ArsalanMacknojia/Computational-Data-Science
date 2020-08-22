# CMPT 353 - Exercise 3
# Arsalan Macknojia

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter


def main(cpu_data):
    # Graph with noise
    plt.figure(figsize=(12, 4))
    plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)

    # LOESS Smoothing
    loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp'], frac=0.025)
    plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')

    # Kalman Smoothing
    kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag(
        [cpu_data['temperature'].std(), cpu_data['cpu_percent'].std(), cpu_data['sys_load_1'].std(),
         cpu_data['fan_rpm'].std()]) ** 2
    transition_covariance = np.diag([0.25, 0.5, 0.05, 0.25]) ** 2
    transition = [[0.97, 0.5, 0.2, -0.001], [0.1, 0.4, 2.2, 0], [0, 0, 0.95, 0], [0, 0, 0, 1]]

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)
    plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')

    # Legend added
    plt.legend(['scatterplot', 'LOESS-smoothed', 'Kalman-smoothed'])
    plt.show()
    plt.title()
    plt.savefig('cpu.svg')


if __name__ == '__main__':
    filename = sys.argv[1]
    cpu_data = pd.read_csv(filename, parse_dates=[0])

    main(cpu_data)
