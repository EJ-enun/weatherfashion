import streamlit as st
import requests
from PIL import Image
import base64
import io
from io import BytesIO
from opencage.geocoder import OpenCageGeocode
from transformers import pipeline, AutoTokenizer
import replicate
import os

#model = AutoModelForCausalLM.from_pretrained("Snowflake/snowflake-arctic-instruct", trust_remote_code=True)
REPLICATE_API_TOKEN='r8_B9vWqzITgJ1KAM11WADipsMESzL91uR0HMLDs'
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

# Set assistant icon to Snowflake logo
icons = {"assistant": "./Snowflake_Logomark_blue.svg", "user": "⛷️"}


os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

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
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything."}]
st.sidebar.button('Clear chat history', on_click=clear_chat_history)

st.sidebar.caption('Built by [Snowflake](https://snowflake.com/) to demonstrate [Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-and-efficient-foundation-language-models-snowflake). App hosted on [Streamlit Community Cloud](https://streamlit.io/cloud). Model hosted by [Replicate](https://replicate.com/snowflake/snowflake-arctic-instruct).')

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Function for generating Snowflake Arctic response
def generate_arctic_response():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
    
    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                           input={"prompt": prompt_str,
                                  "prompt_template": r"{prompt}",
                                  "temperature": temperature,
                                  "top_p": top_p,
                                  }):
        yield str(event)

	
# Load the text-to-caption model
#model = pipeline("text-generation", model="Snowflake/snowflake-arctic-instruct", trust_remote_code=True)

# Load the image-to-text model
model_ydshieh = pipeline('image-to-text', model='ydshieh/vit-gpt2-coco-en')

def main():
  image_url = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/OIG.jpg"
  htp = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/file.png"
  st.image(htp, caption='Dress for the Weather, Impress with Style.')
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

  st.write("Text-to-Caption")
  input_text = st.text_input("Enter your text:")

  for event in replicate.stream(
      "snowflake/snowflake-arctic-instruct",
      input={
          "top_k": 50,
          "top_p": 0.9,
          "prompt": input_text,
          "temperature": 0.2,
          "max_new_tokens": 512,
          "min_new_tokens": 0,
          "stop_sequences": "<|im_end|>",
          "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
          "presence_penalty": 1.15,
          "frequency_penalty": 0.2
      },
  ):
    print(str(event), end="")

  # Store LLM-generated responses
  if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research. Ask me anything."}]

  # Display or clear chat messages
  if st.button("Generate Caption"):
    for message in st.session_state.messages:
      with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

    # result = model(input_text)
    # st.write("Caption:", result)

  # User-provided prompt
  if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="⛷️"):
      st.write(prompt)

   # Generate a new response if last message is not from assistant
   # Generate a new response if last message is not from assistant
  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant", avatar="./Snowflake_Logomark_blue.svg"):
          response = generate_arctic_response()
          full_response = st.write_stream(response)
      message = {"role": "assistant", "content": full_response}
      st.session_state.messages.append(message)









if __name__ == "__main__":
    main()
