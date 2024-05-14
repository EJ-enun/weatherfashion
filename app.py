import streamlit as st
import requests
from PIL import Image
import io
from io import BytesIO
from opencage.geocoder import OpenCageGeocode

st.title('SWEATHER')

# Get the API key from: https://opencagedata.com
key = 'ca22f9473b824f59a109ed0e60d9e551'
geocoder = OpenCageGeocode(key)

address = st.text_input("Enter the location:")
lat = 7.7756663
lng = -72.2214154
if st.button('Get Latitude and Longitude'):
    results = geocoder.geocode(address)
    if results and len(results):
        lat = results[0]['geometry']['lat']
        lng = results[0]['geometry']['lng']
        st.write(f'Latitude: {lat}, Longitude: {lng}')
    else:
        st.write('Location not found')

# Set up your Stable Diffusion Inference Endpoint
INFERENCE_ENDPOINT = "https://api.huggingface.co/models/stable-diffusion/base-1.0/inference"

# Your access token
ACCESS_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": "Bearer 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"}

API_WEATHER = "f0e196555010406d81c233044241305"

#def fetchForecast(lat, lon, apikey):
#    url = "https://api.climacell.co/v3/weather/nowcast"
#    querystring = {"lat":lat,"lon":lon,"unit_system":"si","start_time":"now","fields":"temp,feels_like,weather_code,precipitation,precipitation_type","apikey":apikey}
#    response = requests.request("GET", url, params=querystring)
#    return response.json()

def fetchForecast(lat, lon, apikey):
    url = "http://api.weatherapi.com/v1/current.json"
    querystring = {
        "key": apikey,
        "q": f"{lat},{lon}",
    }
    response = requests.get(url, params=querystring)
    try:
        return response.json()
    except json.JSONDecodeError:
        print("Failed to decode API response")
        print(response.text)
        return {}



def get_precipitation_type(condition_text):
    if "rain" in condition_text.lower():
        return "Rain"
    elif "snow" in condition_text.lower():
        return "Snow"
    else:
        return "Unknown"


def consumeOne(forecast):
    condition_text = forecast["current"]["condition"]["text"]
    #precipitation_type = get_precipitation_type(condition_text)
    return condition_text


#def consumeOne(forecast):
    #condition_text = forecast["current"]["condition"]["text"]
    
    #precipitation_type = get_precipitation_type(condition_text)
    #return print(type(forecast), forecast)
	#{
    #"temp": forecast["current"]["temp_c"],
    #"feel": forecast["current"]["feelslike_c"],
    #"precipitation": forecast["current"]["precip_mm"],
    #"precipitation_type": precipitation_type,
    #"weather_code": forecast["current"]["condition"]["code"],
    #}

def clothing(inp):
    umbrella = inp["is_rainy"] or inp["is_snowy"]
    sunscreen = inp["is_sunny"]
    top = None # Not set yet

    min_temp = min(inp["mintemp"], inp["minfeel"])
    max_temp = max(inp["maxtemp"], inp["maxfeel"])

    if min_temp > 15:
        if max_temp< 25:
            top = "T-Shirt"
        else:
            top = "Tank Top"
    elif max_temp < 5:
        if min_temp > -10:
            top = "Long Sleeves + Coat"
        else:
            top = "Long Sleeves + Sweater"
    else:
        if max_temp - min_temp> 10:
            top = "Long Sleeves + Jacket"
        else:
            top = "Long Sleeves"

    return {"top": top, "sunscreen": sunscreen, "umbrella": umbrella }

#forecasts = fetchForecast(lat, lng, API_WEATHER)
#parsed_forecasts = list(map(consumeOne, forecasts))
parsed_forecasts = consumeOne(fetchForecast(lat, lng, API_WEATHER))
#mintemp = min(list(map(lambda f: f["temp"], parsed_forecasts)))
#maxtemp = max(list(map(lambda f: f["temp"], parsed_forecasts)))

#minfeel = min(list(map(lambda f: f["feel"], parsed_forecasts)))
#maxfeel = max(list(map(lambda f: f["feel"], parsed_forecasts)))

#is_sunny = any(list(map(lambda f: f["weather_code"]=='clear', parsed_forecasts)))
#is_rainy = any(list(map(lambda f: 'rain' in f["weather_code"], parsed_forecasts)))
#is_snowy = any(list(map(lambda f: 'snow' in f["weather_code"], parsed_forecasts)))



def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content


def main():
    st.title("Weather Fashion")
    st.json(consumeOne(fetchForecast(lat, lng, API_WEATHER)))
    # Get user input
    text_prompt = st.text_input("Enter a description for the image:")
    image_bytes = query({"inputs": "Astronaut riding a horse"})
    if st.button("Generate Image"):
        if text_prompt:
            try:
                payload = {"inputs": text_prompt}
                image_data = query(payload)
                #st.write(print(image_bytes))
                image = Image.open(io.BytesIO(image_data))
                #image = Image.open(BytesIO(image_data))
                st.image(image, caption="Generated Image")
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.warning("Please enter a description.")

if __name__ == "__main__":
    main()
