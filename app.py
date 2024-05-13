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
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})


def generate_image_from_text(prompt):
    payload = {
        "text": prompt,
        "num_return_sequences": 1,
        "max_length": 256,
    }
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.post(INFERENCE_ENDPOINT, headers=headers, json=payload)
    image_url = response.json()["generated_images"][0]
    return image_url

def main():
    st.title("Text-to-Image Generator with Streamlit")

    # Get user input
    text_prompt = st.text_input("Enter a description for the image:")
    
    
    if st.button("Generate Image"):
        if text_prompt:
            try:
                image_url = query(text_prompt)
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption="Generated Image")
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.warning("Please enter a description.")

if __name__ == "__main__":
    main()
