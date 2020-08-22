import pandas as pd

def city_with_lowest_precipitation(totals):
    return totals.sum(axis=1).idxmin()

def avg_precipation_per_month(totals, counts):
    return totals.sum(axis=0)/counts.sum(axis=0)

def avg_precipation_per_city(totals, counts):
    return totals.sum(axis=1)/counts.sum(axis=1)

def main():
    totals = pd.read_csv('totals.csv').set_index(keys=['name'])
    counts = pd.read_csv('counts.csv').set_index(keys=['name'])
    print ("City with lowest total precipitation:\n", city_with_lowest_precipitation(totals))
    print ("Average precipitation in each month:\n", avg_precipation_per_month(totals, counts))
    print ("Average precipitation in each city:\n", avg_precipation_per_city(totals, counts))

if __name__ == '__main__':
    main()
