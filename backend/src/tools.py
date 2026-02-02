import httpx
from typing import Dict, Any, List
from src.logger import log_tool_call, log_error

async def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Fetch weather data from Open-Meteo API.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Weather data including temperature, conditions, etc.
    """
    log_tool_call("get_weather", {"lat": latitude, "lon": longitude})
    
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,weather_code,cloud_cover,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            
            # Weather code to description mapping
            weather_codes = {
                0: "Clear sky",
                1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Depositing rime fog",
                51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                77: "Snow grains",
                80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                85: "Slight snow showers", 86: "Heavy snow showers",
                95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
            }
            
            weather_code = current.get("weather_code", 0)
            conditions = weather_codes.get(weather_code, "Unknown")
            
            return {
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "conditions": conditions,
                "precipitation": current.get("precipitation"),
                "rain": current.get("rain"),
                "wind_speed": current.get("wind_speed_10m"),
                "cloud_cover": current.get("cloud_cover"),
                "is_day": current.get("is_day") == 1,
                "latitude": latitude,
                "longitude": longitude
            }
    except Exception as e:
        log_error(e, "get_weather")
        return {"error": str(e)}


async def get_nearby_restaurants(latitude: float, longitude: float, radius_meters: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch nearby restaurants using Overpass API (OpenStreetMap data).
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        radius_meters: Search radius in meters (default: 1000m = 1km)
    
    Returns:
        List of restaurants with name, cuisine, address, etc.
    """
    log_tool_call("get_nearby_restaurants", {"lat": latitude, "lon": longitude, "radius": radius_meters})
    
    try:
        # Overpass API query for restaurants
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Query for amenity=restaurant within radius
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="restaurant"](around:{radius_meters},{latitude},{longitude});
          way["amenity"="restaurant"](around:{radius_meters},{latitude},{longitude});
          relation["amenity"="restaurant"](around:{radius_meters},{latitude},{longitude});
        );
        out body;
        >;
        out skel qt;
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                overpass_url,
                data={"data": query},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            restaurants = []
            for element in data.get("elements", []):
                if element.get("type") in ["node", "way", "relation"]:
                    tags = element.get("tags", {})
                    if tags.get("amenity") == "restaurant":
                        restaurant = {
                            "name": tags.get("name", "Unnamed Restaurant"),
                            "cuisine": tags.get("cuisine", "Not specified"),
                            "address": tags.get("addr:street", ""),
                            "phone": tags.get("phone", ""),
                            "website": tags.get("website", ""),
                            "opening_hours": tags.get("opening_hours", ""),
                            "latitude": element.get("lat"),
                            "longitude": element.get("lon")
                        }
                        restaurants.append(restaurant)
            
            # Limit to 10 most relevant results
            return restaurants[:10]
            
    except Exception as e:
        log_error(e, "get_nearby_restaurants")
        return [{"error": str(e)}]
