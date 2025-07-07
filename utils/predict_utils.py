from models.combined_model import MultiImagePreferenceModel
from utils.image_processing import preprocess_profile_images, analyze_profile_diversity
import base64
import io
from PIL import Image
import numpy as np
import os
import tempfile
import shutil

# Global model instance
preference_model = None

def get_model():
    global preference_model
    if preference_model is None:
        preference_model = MultiImagePreferenceModel()
        try:
            preference_model.load_model()
        except:
            print("No trained model found. Please collect data and train first.")
    return preference_model

def predict_profile_score(images_data, age):
    """
    Predict preference score for profile with multiple images
    """
    model = get_model()
    
    if model.combined_model is None:
        return 0.5  # Default neutral score if no model
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save all images temporarily
        for i, image_data in enumerate(images_data):
            try:
                # Decode and save image
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image = Image.open(io.BytesIO(image_bytes))
                image = image.resize((224, 224))
                image.save(f'{temp_dir}/img_{i+1}.jpg')
            except Exception as e:
                print(f"Error processing image {i+1}: {e}")
        
        # Preprocess images
        processed_images = preprocess_profile_images(temp_dir, model.max_images)
        
        # Get diversity metrics
        diversity = analyze_profile_diversity(temp_dir)
        diversity_vector = [
            diversity['num_faces_detected'],
            diversity['pose_variety'],
            diversity['lighting_variety'],
            diversity['image_quality']
        ]
        
        # Predict
        score = model.predict(processed_images, age, diversity_vector)
        
        return score
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)