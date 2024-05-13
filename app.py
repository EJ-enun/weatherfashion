import streamlit as st
import requests
from PIL import Image
import io
from io import BytesIO
from opencage.geocoder import OpenCageGeocode

st.title('Geocoding App')

# Get the API key from: https://opencagedata.com
key = 'YOUR_OPEN_CAGE_API_KEY'
geocoder = OpenCageGeocode(key)

address = st.text_input("Enter the location:")

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

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content


def main():
    st.title("Weather Fashion")

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
