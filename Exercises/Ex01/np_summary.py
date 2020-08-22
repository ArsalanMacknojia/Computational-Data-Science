import numpy as np

ROW = 0
COLUMN = 1
QUARTER = 4
MONTHS_IN_A_QUARTER = 3


def lowest_precipitation(totals):
    return np.argmin(np.sum(totals, COLUMN))


def avg_precipitation_per_month(totals, counts):
    total_precipitation_per_month = np.sum(totals, ROW)
    total_count = np.sum(counts, ROW)
    return total_precipitation_per_month / total_count


def avg_precipitation_per_city(totals, counts):
    total_precipitation_per_city = np.sum(totals, COLUMN)
    total_count = np.sum(counts, COLUMN)
    return total_precipitation_per_city / total_count


def quarterly_precipitation(totals):
    total_cities = len(totals)
    total_readings = np.size(totals)
    total_reshape_rows = total_readings // MONTHS_IN_A_QUARTER

    array = np.reshape(totals, total_readings)
    quarterly = array.reshape(total_reshape_rows, MONTHS_IN_A_QUARTER)
    quarterly_sum = np.sum(quarterly, COLUMN)
    return quarterly_sum.reshape(total_cities, QUARTER)


def main():
    data = np.load('monthdata.npz')
    totals = data['totals']
    counts = data['counts']

    print("Row with lowest total precipitation:\n", lowest_precipitation(totals))
    print("Average precipitation in each month:\n", avg_precipitation_per_month(totals, counts))
    print("Average precipitation in each city:\n", avg_precipitation_per_city(totals, counts))
    print("Quarterly precipitation totals:\n", quarterly_precipitation(totals))


if __name__ == '__main__':
    main()
