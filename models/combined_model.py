import numpy as np
import cv2
import os
import face_recognition
MAX_IMAGES = 6

def preprocess_images(image):
    """Preprocess a single image for model input."""
    try:
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Image not found or could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        return image
    except Exception as e:
        print(f"Error in preprocess_images: {e}")
        return np.zeros((224, 224, 3))
    
def preprocess_images_folder(folder_path, max_images=MAX_IMAGES):
    """Preprocess all images in a folder. If fewer than max_images, fill with zeros."""
    images = []
    images_file = []
    for i in range(1, max_images + 1):
        image_path = f"{folder_path}/image_{i}.jpg"
        if os.path.exists(image_path):
            images_file.append(image_path)
            
    for image_path in images_file:
        img = preprocess_images(image_path)
        images.append(img)
        
    while len(images) < max_images:
        images.append(np.zeros((224, 224, 3)))
        
    images = images[:max_images]
    return np.array(images)

def extract_features(image_path):
    """Extract facial features from a single image."""
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            return face_encodings[0]
        else:
            print(f"No face found in image: {image_path}")
            return np.zeros(128)
    except Exception as e:
        print(f"Error in extract_feature: {e}")
        return np.zeros(128)
    
def extract_profile_features(folder_path, max_images=MAX_IMAGES):
    """Extract features from all images in a profile folder."""
    all_features = []
    
    for i in range(1, max_images + 1):
        image_path = f"{folder_path}/image_{i}.jpg"
        if os.path.exists(image_path):
            features = extract_features(image_path)
            if features is not None and np.any(features):
                all_features.append(features)

    if not all_features:
        print(f"No valid images found in folder: {folder_path}")
        return np.zeros(128)
    
    return np.mean(all_features, axis=0)

def analyze_profile_diversity(profile_folder, max_images=MAX_IMAGES):
    """
    Analyze diversity of images in a profile
    Returns metrics about pose variety, lighting, etc.
    """
    diversity_metrics = {
        'num_faces_detected': 0,
        'pose_variety': 0.0,
        'lighting_variety': 0.0,
        'image_quality': 0.0
    }
    
    face_landmarks_list = []
    brightness_values = []
    
    for i in range(1, max_images + 1):  # Check up to 6 images
        image_path = f'{profile_folder}/image_{i}.jpg'
        if os.path.exists(image_path):
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Detect faces
                face_landmarks = face_recognition.face_landmarks(image)
                if face_landmarks:
                    diversity_metrics['num_faces_detected'] += 1
                    face_landmarks_list.append(face_landmarks[0])
                
                # Analyze brightness
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
    
    # Calculate pose variety (variation in facial landmarks)
    if len(face_landmarks_list) > 1:
        pose_variations = []
        for i in range(len(face_landmarks_list)):
            for j in range(i+1, len(face_landmarks_list)):
                # Compare nose tip positions as proxy for pose
                nose1 = np.mean(face_landmarks_list[i].get('nose_tip', [[0,0]]), axis=0)
                nose2 = np.mean(face_landmarks_list[j].get('nose_tip', [[0,0]]), axis=0)
                variation = np.linalg.norm(nose1 - nose2)
                pose_variations.append(variation)
        
        diversity_metrics['pose_variety'] = np.mean(pose_variations) if pose_variations else 0.0
    
    # Calculate lighting variety
    if len(brightness_values) > 1:
        diversity_metrics['lighting_variety'] = np.std(brightness_values)
    
    # Overall image quality (based on number of faces detected)
    total_images = len([f for f in os.listdir(profile_folder) if f.endswith('.jpg')])
    if total_images > 0:
        diversity_metrics['image_quality'] = diversity_metrics['num_faces_detected'] / total_images
    
    return diversity_metrics