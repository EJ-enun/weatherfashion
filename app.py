import streamlit as st
import requests
from PIL import Image
from io import BytesIO

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
    image_bytes = query({"inputs": "Astronaut riding a horse",})
    if st.button("Generate Image"):
        if text_prompt:
            try:
                payload = {
                    "text": text_prompt,
                    "num_return_sequences": 1,
                    "max_length": 256,
                }
                #image_data = query(payload)
                st.write(print(image_bytes))
                #image = Image.open(io.BytesIO(image_bytes))
                #image = Image.open(BytesIO(image_data))
                #st.image(image, caption="Generated Image")
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.warning("Please enter a description.")

if __name__ == "__main__":
    main()
