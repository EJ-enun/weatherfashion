# Import necessary libraries
import streamlit as st
import requests
import replicate
from PIL import Image
import base64
import io
from io import BytesIO
from opencage.geocoder import OpenCageGeocode
from transformers import pipeline, AutoTokenizer
import os
import tempfile
import webbrowser


# API keys and tokens are defined
REPLICATE_API_TOKEN='r8_70DiHD1crmyex93p560AlTEP8YLzSjR1AupYr'
key = 'ca22f9473b824f59a109ed0e60d9e551'  # OpenCageData API key
API_WEATHER = "f0e196555010406d81c233044241305"  # WeatherAPI key
ACCESS_TOKEN = "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"  # HuggingFace API key

# Streamlit page configuration is set
st.set_page_config(
    page_title="MeteoroloChic",
    page_icon= "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/filed.png",)

# HuggingFace inference endpoint is defined
INFERENCE_ENDPOINT = "https://api.huggingface.co/models/stable-diffusion/base-1.0/inference"
#API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

headers = {"Authorization": "Bearer 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"}

# Function to get Replicate API token
def get_replicate_api_token():
    os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']

# Function to fetch weather forecast data
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

# Function to get precipitation type
def get_precipitation_type(condition_text):
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

# Function to set background color
def set_background_color(color):
    background_color = f'''
    <style>
    .stApp {{
        background-color: {color};
    }}
    </style>
    '''
    st.markdown(background_color, unsafe_allow_html=True)

# Function to set logo
def set_logo():
    htp = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/file.png"
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        return st.image(htp, caption='Dress for the Weather, Impress with Style.')

# Function to set image load
def set_gen_image_load():
    htp = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/filed.png"
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        return st.image(htp, caption='You are almost done, Keep Going!') 

# Function to consume forecast data
def consumeOne(forecast):
    condition_text = forecast["current"]["condition"]["text"]
    weather_code = forecast["current"]["condition"]["code"]
    feel = forecast["current"]["feelslike_c"]
    precipitation = forecast["current"]["precip_mm"]
    precipitation_type = get_precipitation_type({"condition_text": condition_text})
    return {"condition_text": condition_text, "feels_like": feel, "precipitation": precipitation, "weather_code": weather_code, "precipitation_type":precipitation_type}

# Function to determine clothing based on weather
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

# Function to query HuggingFace model
def query_stable_diff(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

# Function to get location
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



def get_blip_output(inp):
   
   input = {
    "image": inp
}
   output = replicate.run(
    "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    input=input
)
   st.write(f'Salesforce/Blip {output}')
def arctic_gen(weather, options):
	for event in replicate.stream(
    "snowflake/snowflake-arctic-instruct",
    input={
        "top_k": 50,
        "top_p": 0.9,
        "prompt": f"Generate clothes and outfits for {options} in {weather} weather. Do not explain your suggestions, just suggest the clothes. Do not add bullet points or numbers, use only commas as separators",
        "temperature": 0.2,
        "max_new_tokens": 512,
        "min_new_tokens": 0,
        "stop_sequences": "<|im_end|>",
        "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
        "presence_penalty": 1.15,
        "frequency_penalty": 0.2
    },
): yield str(event)
def gen_image_from_arctic_prompt(prompt):
    display_resp(prompt)
 
      
    try:
        payload = {"inputs": display_resp(prompt)}
        image_data = query_stable_diff(payload)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Generated Image")
    except Exception as e:
        st.error(f"Error generating image: {e}")


def address(options):
    address = st.text_input("Address:")
    if st.button('Generate with Arctic'):
      weather = get_location(address)
      st.write(f"Now Let's get you fitted up! This are the weather based outfits generated by the Artic Snowflake Instruct Model")
      set_gen_image_load()
      gen_image_from_arctic_prompt(arctic_gen(weather, options))

	
def wardrobe(selected_options):
    text_prompt = st.text_input("Enter as many fits as you have for this weather in your wardrobe(separate each outfit with a comma):")
    if st.button("Generate Image"):
      if text_prompt:
        get_fits = text_prompt.split(",")
        count_list = len(get_fits)
        model_input = ", ".join(get_fits)
        prompt = f"Create {count_list} separate human {selected_options} wearing {model_input}"
        try:
          payload = {"inputs": prompt}
          image_data = query_stable_diff(payload)
          image = Image.open(io.BytesIO(image_data))
          st.image(image, caption="Generated Image")
        except Exception as e:
          st.error(f"Error generating image: {e}")
      else:
        st.warning("Please enter a description.")


def image_captions(temp, top_p):
  st.write("Captivating Captions: English Language Captions for Your Instagram worthy photos")
  # st.write("Generate a Caption for Every Instagram Picture ")
  uploaded_file = st.file_uploader("Generate a Caption for your Picture", type=["png", "jpg", "jpeg"])
  if uploaded_file is not None:
    data = base64.b64encode(uploaded_file.read()).decode('utf-8')
    img = f"data:application/octet-stream;base64,{data}"
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
      image = image.convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    out_blip = None
    if st.button('Generate'):
      out_blip = get_blip_output(img)
      for event in replicate.stream(
          "snowflake/snowflake-arctic-instruct",
          input={
              "top_k": 50,
              "top_p": top_p,
              "prompt": f" Write a creative caption about only the descriptions made here {out_blip} and do not use the word 'None' in any of your responses. ",
              "temperature": temp,
              "max_new_tokens": 512,
              "min_new_tokens": 0,
              "stop_sequences": "<|im_end|>",
              "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
              "presence_penalty": 1.15,
              "frequency_penalty": 0.2
          },
      ):yield str(event)

def display_resp(event):	    
	
        # Store LLM-generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research."}]

        # Display or clear chat messages
       
        for message in st.session_state.messages:
          with st.chat_message(message["role"]):
            full_response = st.write_stream(event)
            st.session_state.messages = [{"role": "assistant", "content":full_response}]
            st.write(message["content"])
            return message["content"]


  

def reset_app():
  http = "https://weatherfashion.streamlit.app/"
  return webbrowser.open_new_tab(http)

def main():
  st.title('MeteoroloChic')
  st.write("Where Climate Meets Style! Dressing You Right for Every Climate.")
  st.sidebar.button('Reset App', on_click=clear_chat_history)
  #st.subheader("Adjust Photo Caption model parameters")
  temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
  top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
  st.sidebar.caption('Built by Enun Jay at www.devpost.com/enunenun21')
  set_logo()
  set_background_color('#fffbec')
  options = ["Male", "Female", "Non-binary"]
  selected_options = st.multiselect("Choose Gender:", options)
  get_replicate_api_token()
  address(selected_options)
  wardrobe(selected_options)
  display_resp(image_captions(temperature, top_p))
  #stored_caption = store_caption(image_captions(temperature, top_p))
#  show_caption(stored_caption)
if __name__ == "__main__":
    main()
