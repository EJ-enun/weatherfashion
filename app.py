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
#from replicate.client import Client


REPLICATE_API_TOKEN='r8_70DiHD1crmyex93p560AlTEP8YLzSjR1AupYr'
#replicate = Client(api_token=REPLICATE_API_TOKEN)
# API key from: https://opencagedata.com
key = 'ca22f9473b824f59a109ed0e60d9e551'

#API key from: https://weatherapi.com
API_WEATHER = "f0e196555010406d81c233044241305"

#API key from: huggingface.co
ACCESS_TOKEN = "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"

st.title('MeteoroloChic')
st.write("Where Climate Meets Style! Dressing You Right for Every Climate.")


API_URL_ydshieh = "https://api-inference.huggingface.co/models/microsoft/kosmos-2-patch14-224"


# Set up your Stable Diffusion Inference Endpoint
INFERENCE_ENDPOINT = "https://api.huggingface.co/models/stable-diffusion/base-1.0/inference"

# Your access token

API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
headers = {"Authorization": "Bearer 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"}



def get_replicate_api_token():
    os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
    #os.environ['REPLICATE_API_TOKEN'] = 'r8_C0S96Qg9kqLkNy0ilfcOfuRyBPZ2u4f1JTX6P'


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

def set_logo():

    # Relative path to your logo in your GitHub repository
    image_url = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/OIG.jpg"
    htp = "https://raw.githubusercontent.com/EJ-enun/weatherfashion/main/file.png"
    
  

    # Create three columns
    col1, col2, col3 = st.columns([1,6,1])

    # Display the logo in the middle column
    with col2:
        return st.image(htp, caption='Dress for the Weather, Impress with Style.')





def consumeOne(forecast):
    condition_text = forecast["current"]["condition"]["text"]
    weather_code = forecast["current"]["condition"]["code"]
    feel = forecast["current"]["feelslike_c"]
    precipitation = forecast["current"]["precip_mm"]
    precipitation_type = get_precipitation_type({"condition_text": condition_text})
    return {"condition_text": condition_text, "feels_like": feel, "precipitation": precipitation, "weather_code": weather_code, "precipitation_type":precipitation_type}



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
#model_ydshieh = pipeline('image-to-text', model='ydshieh/vit-gpt2-coco-en')
#captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")



def get_blip_output(inp):
   
   input = {
    "image": inp
}
   output = replicate.run(
    "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    input=input
)
   st.write(output)




def main():
  st.sidebar.button('Clear chat history', on_click=clear_chat_history)
  st.sidebar.caption('Built by Enun Jay at www.linkedin.com/in/enun-enun-')
  set_logo()
  set_background_color('#fffbec')
  get_replicate_api_token()
  #load_dotenv()
  
  
  address = st.text_input("Address:")
	
  options = ["Male", "Female", "Non-binary"]
  selected_options = st.multiselect("Choose your options:", options)


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
          "top_p": 0.9,
          "prompt": f" Create an instagram worthy caption from the following text: {out_blip}",
          "temperature": 0.2,
          "max_new_tokens": 512,
          "min_new_tokens": 0,
          "stop_sequences": "<|im_end|>",
          "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
          "presence_penalty": 1.15,
          "frequency_penalty": 0.2
      },
  ):st.write(str(event), end="")
      

    # result = model(input_text)
    # st.write("Caption:", result)

 









if __name__ == "__main__":
    main()
