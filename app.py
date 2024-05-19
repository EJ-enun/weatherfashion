import streamlit as st
import requests
from PIL import Image
import base64
import io
from io import BytesIO
from opencage.geocoder import OpenCageGeocode
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

# Load the Kosmos-2 model
model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")


# API key from: https://opencagedata.com
key = 'ca22f9473b824f59a109ed0e60d9e551'

#API key from: https://weatherapi.com
API_WEATHER = "f0e196555010406d81c233044241305"

#API key from: huggingface.co
ACCESS_TOKEN = "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"

st.title('Climate Couture')
st.write("Where Climate Meets Style! Dressing You Right for Every Climate.")


API_URL_ydshieh = "https://api-inference.huggingface.co/models/microsoft/kosmos-2-patch14-224"


# Set up your Stable Diffusion Inference Endpoint
INFERENCE_ENDPOINT = "https://api.huggingface.co/models/stable-diffusion/base-1.0/inference"

# Your access token

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": "Bearer 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"}



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
    # Extract the 'condition_text' field from the dictionary
    #condition_text = condition_dict.get("condition_text", "")
    if "Rain" in condition_text["condition_text"]:
        return "Rain"
    elif "Snow" in condition_text["condition_text"]:
        return "Snow"
    elif "Sunny" in condition_text["condition_text"]:
        return "Sunny"
    elif "Clear" in condition_text["condition_text"]:
        return "Clear"
    elif "Partly cloudy" in condition_text["condition_text"]:
        return "Partly Cloudy"
    elif "Overcast" in condition_text["condition_text"]:
        return "Overcast"
    else:
        return "Unknown"


def set_background_color(color):
    background_color = f'''
    <style>
    .stApp {{
        background-color: {color};
    }}
    </style>
    '''
    st.markdown(background_color, unsafe_allow_html=True)

def set_logo(logo):

    # Relative path to your logo in your GitHub repository
    logo_path = "path_to_your_logo/logo.png"

    # Create three columns
    col1, col2, col3 = st.columns([1,6,1])

    # Display the logo in the middle column
    with col2:
        return st.image(logo, width=200)





def consumeOne(forecast):
    condition_text = forecast["current"]["condition"]["text"]
    weather_code = forecast["current"]["condition"]["code"]
    feel = forecast["current"]["feelslike_c"]
    precipitation = forecast["current"]["precip_mm"]
    precipitation_type = get_precipitation_type({"condition_text": condition_text})
    return {"condition_text": condition_text, "feels_like": feel, "precipitation": precipitation, "weather_code": weather_code, "precipitation_type":precipitation_type}


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
#parsed_forecasts = consumeOne(fetchForecast(lat, lng, API_WEATHkbshoiej9y0e3uER))

def inputs(inp):
    return
#mintemp = min(list(map(lambda f: f["temp"], parsed_forecasts)))
#maxtemp = max(list(map(lambda f: f["temp"], parsed_forecasts)))

#minfeel = min(list(map(lambda f: f["feel"], parsed_forecasts)))
#maxfeel = max(list(map(lambda f: f["feel"], parsed_forecasts)))

#is_sunny = any(list(map(lambda f: f["weather_code"]=='clear', parsed_forecasts)))
#is_rainy = any(list(map(lambda f: 'rain' in f["weather_code"], parsed_forecasts)))
#is_snowy = any(list(map(lambda f: 'snow' in f["weather_code"], parsed_forecasts)))



def query_stable_diff(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
	
def query_ydshieh(payload):
	response = requests.post(API_URL_ydshieh, headers=headers, json=payload)
	return response.content
	
def get_location(address):
        geocoder = OpenCageGeocode(key)
        results = geocoder.geocode(address)
        if results and len(results):
        	lat = results[0]['geometry']['lat']
        	lng = results[0]['geometry']['lng']
        	st.write(f'Latitude: {lat}, Longitude: {lng}')
        else:
        	st.write('Location not found')
       	return st.json(consumeOne(fetchForecast(lat, lng, API_WEATHER)))


def main():
    image_url = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/OIG.jpg"
    htp="https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/file.png"
    st.image(htp, caption = 'Dress for the Weather, Impress with Style.')
    set_background_color('#fffbec')
    address = st.text_input("Address:")
    weather = None
    options = ["Male", "Female", "Non-binary"]
    selected_options = st.multiselect("Choose your options:", options)
    weather = "Cloudy"
    if st.button('GO'):
        weather = get_location(address)
        st.write(f"Now Let's get you fitted up! Give a detailed description below (color, style, brand) of every clothing which you have that matches the weather.")
    text_prompt = st.text_input("Enter as many fits as you have for this weather in your wardrobe(separate each outfit with a comma):")
    if st.button("Generate Image"):
        if text_prompt:
            get_fits = text_prompt.split(",")
            count_list = len(get_fits)
            model_input = ", ".join(get_fits)
            prompt = f"Create {count_list} separate outfits for {selected_options} based on each description: {model_input}"
            try:
                payload = {"inputs": prompt}
                image_data = query_stable_diff(payload)
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption="Generated Image")
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.warning("Please enter a description.")
    st.write("Fun Image Captioning")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze"):
	    st.write("Model: ", model)
            st.write("Processor: ", processor)
            inputs = processor(images=image, return_tensors="pt", padding=True)
	    st.write("Inputs: ", inputs)
            outputs = model.generate(**inputs)
	    st.write("Outputs: ", outputs)
            result = processor.batch_decode(outputs, skip_special_tokens=True)
            st.write("Prediction:", result)







if __name__ == "__main__":
    main()
