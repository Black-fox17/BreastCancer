import google.generativeai as genai
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key= "AIzaSyC6FhO2BCsCoLp9aYjMnq59mzZ-hkKImi0")
    
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
def get_lung_response(image_data,response):
    image_bytes = BytesIO(image_data)

    image = Image.open(image_bytes)

    prompt = f"""You are a highly knowledgeable pathologist with expertise in histopathology and cancer diagnosis. You will be provided with the following inputs:

    A pathology image: The image is a histopathological sample that may contain evidence of lung cancer.
    Prediction percentages from a trained neural network classifier: These percentages represent the likelihood of the sample belonging to one of three classes:
    Benign
    Adenocarcinoma
    Squamous Cell Carcinoma
    Your task is to:

    Analyze the provided pathology image independently using your knowledge and skills as a pathologist.
    Combine your analysis with the provided prediction percentages to give a more refined and explanatory diagnosis.
    Provide a detailed, professional response explaining:
    Which class is most likely and why.
    The reasoning behind the diagnosis, considering visual cues from the image (e.g., tissue structure, cellular patterns) and the neural network's results.
    Any uncertainty or alternative possibilities and how to approach further investigation or validation.
    Insights or educational context to help a learner or pathologist understand the diagnostic process better.
    Here is the data for your analysis:


    Prediction Percentages:{response}
    Generate your response in clear, well-structured language, ensuring that both experts and learners can understand your explanation."""

    response = model.generate_content([prompt, image])

    return response.text
    

def get_colon_response(image_data,response):
    image_bytes = BytesIO(image_data)

    image = Image.open(image_bytes)

    prompt = f"""You are a highly skilled pathologist with expertise in histopathology and cancer diagnosis. You will be provided with the following inputs:

    A pathology image: The image is a histopathological sample of colon tissue that may exhibit evidence of cancer.
    Prediction percentages from a trained neural network classifier: These percentages represent the likelihood of the sample belonging to one of two classes:
    Adenocarcinoma
    Benign
    Your responsibilities are to:

    Independently analyze the pathology image using your expertise in histopathology, focusing on visual indicators such as glandular structures, cellular abnormalities, and tissue architecture.
    Integrate your analysis with the prediction percentages provided by the classifier to form a refined, well-balanced diagnosis.
    Generate a detailed explanation, including:
    The most likely class and the reasoning behind your conclusion.
    Observations from the image (e.g., signs of malignancy such as irregular glandular formation or the absence of invasive features in benign samples).
    A discussion of uncertainties or alternative considerations, if applicable, and recommendations for further validation or tests.
    Educational insights to help the user (e.g., a pathologist or a learner) understand the diagnostic process and interpret the results effectively.
    Here is the input data for your analysis:

    
    Prediction Percentages:{response}
    Compose your response in a professional, clear, and concise manner, ensuring it is accessible to both experts and learners. Use this opportunity to educate the user while providing a highly accurate and meaningful diagnostic assessment."""

    response = model.generate_content([prompt, image])

    return response.text