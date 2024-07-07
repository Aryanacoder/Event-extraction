from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def location_to_coordinates(location):
    # Initialize the Nominatim geocoder with a unique user-agent
    geolocator = Nominatim(user_agent="your_unique_user_agent")

    try:
        # Attempt to get the location data
        location_data = geolocator.geocode(location, timeout=10)
        if location_data:
            return (location_data.latitude, location_data.longitude)
        else:
            return "Location not found"
    except GeocoderTimedOut:
        return "Geocoding service timed out"
    except GeocoderServiceError as e:
        return f"Geocoding service error: {e}"

# Example usage:
location_name = "bangladesh"
coordinates = location_to_coordinates(location_name)
print(f"The coordinates of {location_name} are: {coordinates}")
