import google.generativeai as genai
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key= "AIzaSyC6FhO2BCsCoLp9aYjMnq59mzZ-hkKImi0")
    
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
def get_response(image_data,response):
    image_bytes = BytesIO(image_data)

    image = Image.open(image_bytes)

    prompt = f"""You are a highly knowledgeable pathologist with extensive expertise in histopathology and breast cancer diagnosis. You will be provided with the following inputs:

        A pathology image: This histopathological sample is from breast tissue and is intended for cancer classification and diagnosis.
        Classification results from a trained neural network: These results include:
        The predicted class based on the highest probability.
        The probability distribution for all possible classes in the dataset.
        Raw probability values for reference.
        Your responsibilities are to:

        Analyze the pathology image independently using your expertise, focusing on key visual features such as glandular structures, cellular organization, mitotic activity, and any signs of malignancy or specific breast cancer subtypes.
        Combine your analysis with the provided classification results to refine the diagnostic interpretation, considering both the predicted class and the probabilities of alternative classes.
        Generate a professional, detailed response that includes:
        The most likely diagnosis and reasoning based on both the image and the neural network results.
        A discussion of the confidence in the prediction, explaining why the predicted class is likely or unlikely based on probabilities and image features.
        Considerations for alternative diagnoses if probabilities for other classes suggest some uncertainty or overlap.
        Recommendations for further steps, such as additional tests or analyses if needed.
        Educational insights to explain the diagnostic process, aiding learners or users in understanding how predictions are made and verified.
        Here is the input data for your analysis:

        Result:{response}
        Compose your response in a clear, professional tone, suitable for pathologists and learners. Provide meaningful context for your conclusions, explaining how both the classifier and your analysis support the diagnosis."""
    response = model.generate_content([prompt, image])

    return response.text
    