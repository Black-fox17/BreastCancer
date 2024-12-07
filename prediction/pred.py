import joblib
import google.generativeai as genai
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from huggingface_hub import from_pretrained_keras
import matplotlib.pyplot as plt
from huggingface_hub import login
import google.generativeai as genai


genai.configure(api_key = genai_api_key)#put your genai api key here
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

clf_pipeline = joblib.load("trained_model.pkl")
login(token= huggingface_key)# Your huggingface token

# from patch_embedding_types import EmbeddingResult

loaded_model = from_pretrained_keras("google/path-foundation")

# Call inference
infer = loaded_model.signatures["serving_default"]

def load_as_embeddings(batches, target_size=(224, 224)):
  embeddings = []
  # Load the image
  for image_path in batches:
    image_byte = tf.io.read_file(image_path)
    image = Image.open(BytesIO(image_byte.numpy())).convert("RGB")  # Ensure 3-channel RGB
    image = image.resize(target_size, Image.Resampling.LANCZOS)  # Resize the image
    image_array = np.array(image)

    # Convert to a tensor and normalize
    tensor = tf.cast(tf.expand_dims(image_array, axis=0), tf.float32) / 255.0
    embedding = infer(tf.constant(tensor))

    embeddings.append(embedding)
  print(f"Done loading tensors into embeddings length: {len(embeddings)}")

  return embeddings

def get_embeddings(
    embedding_results,
) -> np.ndarray:
  """Returns numpy array of embeddings returned in embedding results list."""
  return np.array([e['output_0'].numpy().flatten() for e in embedding_results])

def get_predictions(sample_image):
  embedding = load_as_embeddings([sample_image])
  total_test = get_embeddings(embedding)
  predictions = clf_pipeline.predict_proba(total_test)
  return predictions

def get_response(image_path):
  predictions = get_predictions(image_path)#The output explained belpw
  prompt = f"""You are a highly specialized medical AI system designed to assist in diagnosing breast cancer by analyzing images and predictive outputs from a trained image classifier. You will receive:

  1. An input image of a breast scan (X-ray or mammogram).
  2. A list of two probabilities representing the classifier's predictions:{predictions}
    The probability of the tumor being benign.
    The probability of the tumor being malignant.
  Your task is to carefully evaluate both the image and the given probabilities to make a more refined judgment about the classification. Use the image to validate or adjust the initial prediction, considering intricate visual patterns that might have been overlooked. Then:

  Provide a final classification as either "Benign" or "Malignant."
  Offer a brief explanation for your decision, referencing both the image features and the probabilities provided.
  Include a confidence score for your final judgment (ranging from 0% to 100%).
  Ensure your response is clear, medically accurate, and supports further clinical analysis."""
  image = Image.open(image_path)
  response = model.generate_content([prompt,image])
  return response.text

result = get_response("path")#Only image path (png|jpg or any other image types not dcm)
print(result)