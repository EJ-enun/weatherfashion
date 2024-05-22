# MeteoroloCHic
#Weather Fashion for people who want to look stylish in all weathers.

This readme describes MeteoroloChic, a Streamlit application that helps you dress for the weather and generate captions for your outfit photos!

**What it does:**

* **Gets your weather:** Enter your address and MeteoroloChic will fetch the current weather conditions for your location.
* **Recommends clothing:** Based on the weather, MeteoroloChic will suggest clothing items that are appropriate for the temperature and conditions (e.g., rain, sun).
* **Generates outfit images:** Describe your outfit and MeteoroloChic will use a powerful AI model to generate an image of someone wearing your described outfit.
* **Creates captions:** Upload a photo of your outfit and MeteoroloChic will generate a social media caption for the photo based on the image content.

**How to use it:**

1. Open MeteoroloChic in your web browser.
2. Enter your address in the "Address" section and click "GO" to get your weather.
3. In the "Choose Gender" section, select your gender or non-binary.
4. In the "Wardrobe" section, describe your outfit (separate each item with a comma) and click "Generate Image" to see an image of someone wearing your outfit.
5. In the "Captivating Captions" section, upload a photo of your outfit and click "Generate" to create a caption for your photo.

**Additional features:**

* You can adjust the temperature and a parameter called "top_p" in the sidebar to influence the style of the generated captions.
* You can reset the app to its initial state by clicking the "Reset App" button in the sidebar.

**Technical details:**

MeteoroloChic uses several advanced technologies:

* Streamlit: A framework for building web apps in Python.
* OpenCageData API: To get your location coordinates from your address.
* WeatherAPI: To get the current weather data.
* Hugging Face API: To generate captions for your photos.
* Replicate: To generate images of people wearing your outfit descriptions.

I hope you find MeteoroloChic helpful for dressing for the weather and creating stylish social media posts!

