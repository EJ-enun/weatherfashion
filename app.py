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
import random
import pyperclip

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
    # API endpoint
    url = "http://api.weatherapi.com/v1/current.json"
    
    # Parameters for the API request
    querystring = {
        "key": apikey,  # API key
        "q": f"{lat},{lon}",  # Latitude and Longitude
    }
    
    # Sending a GET request to the API
    response = requests.get(url, params=querystring)
    
    try:
        # Attempt to return the response in JSON format
        return response.json()
    except json.JSONDecodeError:
        # If there's an error in decoding the JSON, print an error message and the response text
        print("Failed to decode API response")
        print(response.text)
        
        # Return an empty dictionary in case of an error
        return {}

# Function to get precipitation type
def get_precipitation_type(condition_text):
    # Check if the condition text contains "Rain"
    if "Rain" in condition_text["condition_text"]:
        return "Rain"  # Return "Rain" if true
    # Check if the condition text contains "Snow"
    elif "Snow" in condition_text["condition_text"]:
        return "Snow"  # Return "Snow" if true
    # Check if the condition text contains "Sunny"
    elif "Sunny" in condition_text["condition_text"]:
        return "Sunny"  # Return "Sunny" if true
    # Check if the condition text contains "Clear"
    elif "Clear" in condition_text["condition_text"]:
        return "Clear"  # Return "Clear" if true
    # Check if the condition text contains "Partly cloudy"
    elif "Partly cloudy" in condition_text["condition_text"]:
        return "Partly Cloudy"  # Return "Partly Cloudy" if true
    # Check if the condition text contains "Overcast"
    elif "Overcast" in condition_text["condition_text"]:
        return "Overcast"  # Return "Overcast" if true
    else:
        return "Unknown"  # If none of the above conditions are met, return "Unknown"


# Function to set background color
def set_background_color(color):
    # Define a string that contains CSS to change the background color
    background_color = f'''
    <style>
    .stApp {{  # Target the Streamlit application
        background-color: {color};  # Set the background color
    }}
    </style>
    '''
    # Use Streamlit's markdown method to apply the CSS
    # The 'unsafe_allow_html=True' argument allows the use of HTML in the markdown
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

# This function consumes forecast data and extracts relevant information.
def consumeOne(forecast):
    # Extracts the text description of the current weather condition from the forecast data
    condition_text = forecast["current"]["condition"]["text"]
    
    # Extracts the weather code corresponding to the current weather condition from the forecast data
    weather_code = forecast["current"]["condition"]["code"]
    
    # Extracts the current 'feels like' temperature (in Celsius) from the forecast data
    feel = forecast["current"]["feelslike_c"]
    
    # Extracts the current precipitation (in mm) from the forecast data
    precipitation = forecast["current"]["precip_mm"]
    
    # Determines the type of precipitation (e.g., rain, snow, etc.) based on the condition text
    precipitation_type = get_precipitation_type({"condition_text": condition_text})
    
    # Returns a dictionary containing the extracted information
    return {"condition_text": condition_text, "feels_like": feel, "precipitation": precipitation, "weather_code": weather_code, "precipitation_type":precipitation_type}


# This function determines the appropriate clothing based on weather conditions.
def clothing(inp):
    # Determines if an umbrella is needed based on whether it's rainy or snowy
    umbrella = inp["is_rainy"] or inp["is_snowy"]
    
    # Determines if sunscreen is needed based on whether it's sunny
    sunscreen = inp["is_sunny"]
    
    # Initializes the 'top' variable which will hold the appropriate clothing for the upper body
    top = None
    
    # Determines the minimum temperature by comparing the minimum temperature and the 'feels like' temperature
    min_temp = min(inp["mintemp"], inp["minfeel"])
    
    # Determines the maximum temperature by comparing the maximum temperature and the 'feels like' temperature
    max_temp = max(inp["maxtemp"], inp["maxfeel"])
    
    # Determines the appropriate clothing based on the temperature range
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
    
    # Returns a dictionary containing the determined clothing and whether sunscreen and an umbrella are needed
    return {"top": top, "sunscreen": sunscreen, "umbrella": umbrella }


# Function to query HuggingFace model
def query_stable_diff(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

# Function to get location
def get_location(address):
    # Create a geocoder using OpenCageGeocode with the provided key
    geocoder = OpenCageGeocode(key)
    
    # Use the geocoder to get the results of the geocoding of the address
    results = geocoder.geocode(address)
    
    # If there are results and the results list is not empty
    if results and len(results):
        # Get the latitude and longitude from the first result
        lat = results[0]['geometry']['lat']
        lng = results[0]['geometry']['lng']
        
        # Write the latitude and longitude to the Streamlit app
        st.write(f'Latitude: {lat}, Longitude: {lng}')
    else:
        # If no results were found, write that the location was not found
        st.write('Location not found')
    
    # Return the JSON response of the forecast for the latitude and longitude
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
def arctic_gen(weather, options_r, options_g):
	for event in replicate.stream(
    "snowflake/snowflake-arctic-instruct",
    input={
        "top_k": 50,
        "top_p": 0.9,
	"prompt": f"Generate an outfit for a {options_r} {options_g} in {weather} weather. Do not explain your suggestions, just suggest the clothes and make only one suggestion, and do not use separators like numbers or bullet points.",
        #"prompt": f"Generate 5 clothes and outfits for a {options_r} {options_g} in {weather} weather. Do not explain your suggestions, just suggest the clothes. use numbers as separators",
        "temperature": 0.2,
        "max_new_tokens": 512,
        "min_new_tokens": 0,
        "stop_sequences": "<|im_end|>",
        "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
        "presence_penalty": 1.15,
        "frequency_penalty": 0.2
    },
): yield str(event)

def get_random_resp(prompt):
  st.write(prompt)
  prompt = display_resp(prompt)
  # Split the string into suggestions based on the digit followed by a dot and space
  suggestions = prompt.split(r' \d\. ')
    
  # Remove the first number from the first suggestion
  suggestions[0] = suggestions[0][3:]

  # Store the suggestions in a dictionary
  responses = {i: suggestion for i, suggestion in enumerate(suggestions, 1)}

  # Iterate through the dictionary and randomly pick one of the responses
  for _ in range(5):
    selected_response = random.choice(list(responses.values()))
    return selected_response
  

  

# Function to generate an image from a prompt
def gen_image_from_arctic_prompt(prompt):
    # Display the response of the prompt on the Streamlit app
    st.write(f"This is the response - {display_resp(prompt)}") 
    
    try:
        # Create a payload with the response of the prompt as inputs
        payload = {"inputs": display_resp(prompt) }
        
        # Query the stable difference function with the payload and get the image data
        image_data = query_stable_diff(payload)
        
        # Open the image data as an image
        image = Image.open(io.BytesIO(image_data))
        
        # Display the image on the Streamlit app with a caption
        st.image(image, caption="Generated Image")
    except Exception as e:
        # If there is an error in generating the image, display the error on the Streamlit app
        st.error(f"Error generating image: {e}")


# Function to generate an image from a prompt
def address(options_r, options_g):
    # Create a text input field in the Streamlit app for the user to enter an address
    address = st.text_input("Address:")
    
    # If the user clicks the 'Generate with Arctic' button
    if st.button('Generate with Arctic'):
        # Get the weather for the entered address
        weather = get_location(address)
        
        # Display a message on the Streamlit app
        st.write(f"Now Let's get you fitted up! These are the weather-based outfits generated by the Arctic Snowflake Instruct Model")
        
        # Call the function to set the image load
        set_gen_image_load()
        
        # Generate an image from the Arctic prompt with the weather and the options
        gen_image_from_arctic_prompt(arctic_gen(weather, options_r, options_g))

	
# Function to generate an image from a prompt
def wardrobe(options_r, options_g):
    # Create a text input field in the Streamlit app for the user to enter outfits
    text_prompt = st.text_input("Enter as many fits as you have for this weather in your wardrobe(separate each outfit with a comma):")
    
    # If the user clicks the 'Generate Image' button
    if st.button("Generate Image"):
        # If the user has entered a description
        if text_prompt:
            # Split the entered outfits by commas
            get_fits = text_prompt.split(",")
            
            # Count the number of outfits
            count_list = len(get_fits)
            
            # Join the outfits with commas
            model_input = ", ".join(get_fits)
            
            # Create a prompt for the model
            prompt = f"Create {count_list} separate outfits for a {options_r} {options_g}  wearing {model_input}"
            
            try:
                # Create a payload with the prompt as inputs
                payload = {"inputs": prompt}
                
                # Query the stable difference function with the payload and get the image data
                image_data = query_stable_diff(payload)
                
                # Open the image data as an image
                image = Image.open(io.BytesIO(image_data))
                
                # Display the image on the Streamlit app with a caption
                st.image(image, caption="Generated Image")
            except Exception as e:
                # If there is an error in generating the image, display the error on the Streamlit app
                st.error(f"Error generating image: {e}")
        else:
            # If the user has not entered a description, display a warning
            st.warning("Please enter a description.")

# Function to copy text to clipboard
def copy(text):
   
        # Create a button labeled "Copy to clipboard"
        copy_to_clipboard = st.button(label="Copy to clipboard :clipboard:")
        
        # If the button is clicked
        if copy_to_clipboard:
            try:
                # Try to copy the provided text to the clipboard
                pyperclip.copy(text)
            except pyperclip.PyperclipException:
                # If an exception occurs, display a warning message
                st.warning("Copy Exception Thrown")


# Function to generate captions for an image
def image_captions(temp, top_p):
    # Display a title on the Streamlit app
    st.write("Captivating Captions: English Language Captions for Your Instagram worthy photos")
    
    # Create a file uploader in the Streamlit app for the user to upload an image
    uploaded_file = st.file_uploader("Generate a Caption for your Picture", type=["png", "jpg", "jpeg"])
    
    # If the user has uploaded a file
    if uploaded_file is not None:
        # Read the uploaded file and encode it in base64
        data = base64.b64encode(uploaded_file.read()).decode('utf-8')
        
        # Create a data URL for the image
        img = f"data:application/octet-stream;base64,{data}"
        
        # Open the uploaded file as an image
        image = Image.open(uploaded_file)
        
        # If the image is in RGBA mode, convert it to RGB mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Display the image on the Streamlit app with a caption
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Initialize the output variable
        out_blip = None
        
        # If the user clicks the 'Generate' button
        if st.button('Generate'):
            # Call the function to get the blip output for the image
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

# Function to display responses
def display_resp(event):	    
    # If "messages" is not in the session state keys
    if "messages" not in st.session_state.keys():
        # Initialize "messages" in the session state with a welcome message
        st.session_state.messages = [{"role": "assistant", "content": "Hi. I'm Arctic, a new, efficient, intelligent, and truly open language model created by Snowflake AI Research."}]

    # For each message in the session state messages
    for message in st.session_state.messages:
        # Create a chat message with the role of the message
        with st.chat_message(message["role"]):
            # Write the event to the Streamlit app and store the full response
            full_response = st.write_stream(event)
            
            # Update the session state messages with the full response
            st.session_state.messages = [{"role": "assistant", "content":full_response}]
            
            # Write the content of the message to the Streamlit app
            st.write(message["content"])
		
		
            #Copy the text
            copy(full_response)
		
            # Return the content of the message
            return message["content"]


  

def reset_app():
  http = "https://weatherfashion.streamlit.app/"
  return webbrowser.open_new_tab(http)
# Function to add dropdowns to the Streamlit app
def add_dropdowns():
    # Define the race and gender options
    race_options = ["Asian", "Black", "Biracial",  "White"]
    gender_options = ["Male", "Female", "Non-binary"]
    
    # Add a multiselect for the user to choose a gender
    gender_selected_option = st.multiselect("Choose Gender:", gender_options)
    
    # Add a selectbox for the user to choose a race
    race_selected_option = st.selectbox('Select an option:', race_options)
    
    # Return the selected race and gender options
    return [race_selected_option, gender_selected_option]
# Main function
def main():
    # Set the title of the Streamlit app
    st.title('MeteoroloChic')
    
    # Write a description on the Streamlit app
    st.write("Where Climate Meets Style! Dressing You Right for Every Climate.")
    
    # Add a button to the sidebar to reset the app
    st.sidebar.button('Reset App', on_click=clear_chat_history)
    
    # Add sliders to the sidebar to adjust the temperature and top_p parameters
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    
    # Add a caption to the sidebar
    st.sidebar.caption('Built by Enun Jay at www.devpost.com/enunenun21')
    
    # Call the function to set the logo
    set_logo()
    
    # Call the function to set the background color
    set_background_color('#fffbec')
    
    # Define the race and gender options
    race_options = ["Asian", "Black", "Biracial",  "White"]
    gender_options = ["Male", "Female", "Non-binary"]
    
    # Add a multiselect for the user to choose a gender
    gender_selected_option = st.multiselect("Choose Gender:", gender_options)
    
    # Add a selectbox for the user to choose a race
    race_selected_option = st.selectbox('Choose Race:', race_options)
    
    # Call the function to get the Replicate API token
    get_replicate_api_token()
    
    # Call the function to get the address
    address(race_selected_option,gender_selected_option)
    
    # Call the function to get the wardrobe
    wardrobe(race_selected_option, gender_selected_option)
    
    # Call the function to display the response of the image captions
    display_resp(image_captions(temperature, top_p))
#  show_caption(stored_caption)
if __name__ == "__main__":
    main()
