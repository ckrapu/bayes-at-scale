'''
Functionality for manipulating and linking locations in geographic space.
'''

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


