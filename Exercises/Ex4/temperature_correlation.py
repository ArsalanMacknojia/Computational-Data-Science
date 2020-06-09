# CMPT 353 - Exercise 4
# Arsalan Macknojia


import sys
import numpy as np
import pandas as pd
from math import radians
import matplotlib.pyplot as plt

EARTH_RADIUS_IN_KMS = 6372.8
SQUARE_M_TO_SQUARE_KMS = 1 / 1000000


def distance(city, station):
    """
    This function implements Haversine formula to calculate distance between two latitude and longitude coordinates.
    Reference: https://rosettacode.org/wiki/Haversine_formula

    :param city: pd.series - contain city data (name, population area, latitude, longitude)
    :param station: DataFrame - contain weather stations data (observations, avg_tmax, station, latitude, longitude, elevation)
    :return: pd.series - contain distance in kms between the city and weather stations.
    """
    city_lat = city.get(key='latitude')
    city_lon = city.get(key='longitude')
    station_lat = station['latitude']
    station_lon = station['longitude']

    lat_diff = np.subtract(station_lat, city_lat).apply(radians)
    lon_diff = np.subtract(station_lon, city_lon).apply(radians)
    city_lat = radians(city_lat)
    stations_lat = station_lat.apply(radians)
    a = np.sin(lat_diff / 2) ** 2 + np.cos(city_lat) * np.cos(stations_lat) * np.sin(lon_diff / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_IN_KMS * c


def best_tmax(city, stations):
    """
    This function finds the closest weather station for the given city and return station's avg_tmax value.
    :param city: pd.series - contain city data (name, population area, latitude, longitude)
    :param stations: DataFrame - contain weather stations data (observations, avg_tmax, station, latitude, longitude, elevation)
    :return: float - avg_tmax value
    """
    city_stations_distance = distance(city, stations)
    closest_station = stations.iloc[np.argmin(city_stations_distance)]
    return closest_station["avg_tmax"]


def plot_graph(cities, output_file):
    plt.plot(cities['avg_tmax'], cities['population_density'], 'b.')
    plt.title('Temperature vs Population Density')
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.savefig(output_file)


def main():
    # Get cities and stations data from the given files.
    stations_file = sys.argv[1]
    city_data_file = sys.argv[2]
    output_file = sys.argv[3]

    cities = pd.read_csv(city_data_file)
    stations = pd.read_json(stations_file, lines=True)

    # Filter data
    cities = cities.dropna()
    cities['area sq km'] = cities['area'] * SQUARE_M_TO_SQUARE_KMS
    cities = cities[cities['area sq km'] <= 10000]

    # Calculate population density (population / area km²)
    cities['population_density'] = cities['population'] / cities['area sq km']

    # Calculate average max temperature for each city.
    cities_avg_tmax = cities.apply(best_tmax, args=(stations,), axis=1)
    cities['avg_tmax'] = cities_avg_tmax / 10

    plot_graph(cities, output_file)


if __name__ == '__main__':
    main()
