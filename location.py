'''
Functionality for manipulating and linking locations in geographic space.
'''

import numpy as np
import requests

def get_lat_long(address, api_key):
    '''
    Given an address, return the latitude and longitude using the Google Maps API.
    Note that this is pretty expensive, with 1K calls costing $5.

    Parameters
    ----------
    address : str
        The address to geocode. Works best if it's something including city, state, and country.

    api_key : str
        The Google Maps API key. This is required to make the call. This needs to have the Geocoding API enabled.

    Returns
    -------
    lat : float
        The latitude of the address.
    
    lng : float
        The longitude of the address.
    '''
    
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "OK":
            location = result["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            # Print out information regarding the failure
            print(result["status"])            

            if 'error_message' in result:
                print(result["error_message"])

            return None, None
    else:
        return None, None


def find_components(adj_matrix):
    """Find disconnected components in the graph."""
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    def dfs(v, comp):
        visited[v] = True
        comp.append(v)
        for u in adj_matrix[v].nonzero()[1]:
            if not visited[u]:
                dfs(u, comp)

    for i in range(n):
        if not visited[i]:
            comp = []
            dfs(i, comp)
            components.append(comp)

    return components

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points on the earth."""
    R = 6371  # Earth radius in kilometers

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def shortest_connection(city_df, components):
    """Find the shortest connection between components using Haversine distance."""
    min_distance = float('inf')
    edge_to_add = None

    for i, comp1 in enumerate(components):
        for j, comp2 in enumerate(components):
            if i >= j:
                continue
            for city1 in comp1:
                for city2 in comp2:
                    lat1, lon1 = city_df.loc[city1, ['lat', 'long']]
                    lat2, lon2 = city_df.loc[city2, ['lat', 'long']]
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    if dist < min_distance:
                        min_distance = dist
                        edge_to_add = (city1, city2)

    return edge_to_add

def connect_components(city_df, adj_matrix):
    """Connect all components in the graph."""
    components = find_components(adj_matrix)
    while len(components) > 1:
        edge = shortest_connection(city_df, components)
        adj_matrix[edge[0], edge[1]] = adj_matrix[edge[1], edge[0]] = 1
        components = find_components(adj_matrix)

    return adj_matrix