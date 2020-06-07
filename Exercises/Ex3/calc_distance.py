# CMPT 353 - Exercise 3
# Arsalan Macknojia

import sys
import pandas as pd
import numpy as np
from xml.dom import minidom
from math import radians, sin, cos, sqrt, asin
from pykalman import KalmanFilter

KMS_TO_METERS = 1000
EARTH_RADIUS_IN_KMS = 6372.8

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def get_data(filename):
    """
    Extract latitude and longitude information from the given XML and package it in a data frame.
    :param filename: str - XML filename
    :return: DataFrame
    """
    xml = minidom.parse(filename)
    coordinates = xml.getElementsByTagName("trkpt")

    latitude = []
    longitude = []
    for coordinate in coordinates:
        latitude.append(coordinate.getAttribute("lat"))
        longitude.append(coordinate.getAttribute("lon"))

    return pd.DataFrame(data={'lat': latitude, 'lon': longitude}, dtype=float)


def distance(df):
    """
    This function take latitude/longitude points as DataFrame and calculate total distance travelled using haversince formula.
    Reference:
    1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
    2. https://rosettacode.org/wiki/Haversine_formula

    :param df: DataFrame - coordinates as latitude and longitude
    :return: float - distance in meters
    """

    # Update DF to have adjacent points into the same rows (lon, lat, lon1, lat1)
    adj_points = df.shift(-1).rename(columns={"lon": "lon1", "lat": "lat1"})
    df = df.join(adj_points)

    # Calculate distance between adjacent points and sum it up to get the total distance.
    adj_points_dist = df.apply(haversine, axis=1)
    adj_points_dist = adj_points_dist[np.logical_not(np.isnan(adj_points_dist))]
    total_dist = np.sum(adj_points_dist) * KMS_TO_METERS

    return total_dist


def haversine(coordinates):
    """
    This function implements Haversine formula to calculate distance between two latitude and longitude coordinates.
    Reference: https://rosettacode.org/wiki/Haversine_formula

    :param series: array - adjacent points coordinates
    :return: float - distance in two coordinates in Kms
    """
    lat1 = coordinates.get(key='lat')
    lat2 = coordinates.get(key='lat1')
    lon1 = coordinates.get(key='lon')
    lon2 = coordinates.get(key='lon1')

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return EARTH_RADIUS_IN_KMS * c


def smooth(df):
    """
    This function apply Kalman smoothing on the given data.
    :param df: DataFrame
    :return: DataFrame
    """

    # No prior knowledge of where the walk started. Taking first data point as the starting point.
    initial_state = df.iloc[0]
    # GPS is accurate to about 5 metres, however, in reality it can to be several times that: 15 or 20 metres.
    # Hence taking 10 meters as the mean standard deviation. (Note: 1 degree of latitude or longitude is about 10^5 meters)
    observation_covariance = np.diag([0.00010, 0.00010]) ** 2
    # No prior knowledge to predict the next coordinates hence using the default transition matrix.
    transition_matrix = np.diag([1,1])
    transition_covariance = np.diag([0.00010, 0.00010]) ** 2

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        observation_covariance=observation_covariance,
        transition_matrices=transition_matrix,
        transition_covariance=transition_covariance
    )
    kalman_smoothed, _ = kf.smooth(df)
    return pd.DataFrame(data={'lat': kalman_smoothed[:, 0], 'lon': kalman_smoothed[:, 1]}, dtype=float)


def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))

    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
