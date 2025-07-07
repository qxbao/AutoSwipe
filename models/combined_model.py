import tensorflow as tf
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MultiImagePreferenceModel:
    def __init__(self, max_images=6):
        self.max_images = max_images
        self.image_model = None
        self.combined_model = None
        self.scaler = StandardScaler()
        
    def build_image_model(self):
        """
        Build CNN for processing multiple images per profile
        """
        # Base model for single image
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Input for multiple images
        multi_image_input = tf.keras.Input(shape=(self.max_images, 224, 224, 3))
        
        # Process each image through the same base model
        image_features = []
        for i in range(self.max_images):
            img = tf.keras.layers.Lambda(lambda x, idx=i: x[:, idx, :, :, :])(multi_image_input)
            features = base_model(img, training=False)
            features = tf.keras.layers.GlobalAveragePooling2D()(features)
            image_features.append(features)
        
        # Stack features from all images
        stacked_features = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(image_features)
        
        # Use attention mechanism to weight different images
        attention_weights = tf.keras.layers.Dense(1, activation='softmax')(stacked_features)
        attention_weights = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(attention_weights)
        
        # Weighted average of image features
        weighted_features = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x[0] * tf.expand_dims(x[1], axis=-1), axis=1)
        )([stacked_features, attention_weights])
        
        # Additional processing
        x = tf.keras.layers.Dense(256, activation='relu')(weighted_features)
        x = tf.keras.layers.Dropout(0.3)(x)
        final_image_features = tf.keras.layers.Dense(128, activation='relu')(x)
        
        self.image_model = tf.keras.Model(multi_image_input, final_image_features)
        return self.image_model
    
    def build_combined_model(self):
        """
        Build combined model for multiple images + demographics + diversity metrics
        """
        # Multi-image input
        image_input = tf.keras.Input(shape=(self.max_images, 224, 224, 3), name='images')
        image_features = self.image_model(image_input)
        
        # Demographics input (age)
        demo_input = tf.keras.Input(shape=(1,), name='demographics')
        demo_features = tf.keras.layers.Dense(32, activation='relu')(demo_input)
        
        # Diversity metrics input
        diversity_input = tf.keras.Input(shape=(4,), name='diversity')  # 4 diversity metrics
        diversity_features = tf.keras.layers.Dense(16, activation='relu')(diversity_input)
        
        # Combine all features
        combined = tf.keras.layers.concatenate([image_features, demo_features, diversity_features])
        x = tf.keras.layers.Dense(128, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        
        # Output preference score
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='preference')(x)
        
        self.combined_model = tf.keras.Model(
            inputs=[image_input, demo_input, diversity_input], 
            outputs=output
        )
        
        self.combined_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )
        
        return self.combined_model
    
    def load_training_data(self):
        """
        Load profile data from database for training
        """
        conn = sqlite3.connect('data/swipes.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT profile_folder, age, preference_score FROM profiles')
        data = cursor.fetchall()
        conn.close()
        
        if len(data) < 20:
            raise ValueError("Not enough training data. Need at least 20 profiles.")
        
        from utils.image_processing import preprocess_profile_images, analyze_profile_diversity
        
        images = []
        ages = []
        diversity_metrics = []
        scores = []
        
        for profile_folder, age, score in data:
            try:
                # Load and preprocess multiple images
                profile_images = preprocess_profile_images(profile_folder, self.max_images)
                
                # Get diversity metrics
                diversity = analyze_profile_diversity(profile_folder)
                diversity_vector = [
                    diversity['num_faces_detected'],
                    diversity['pose_variety'],
                    diversity['lighting_variety'],
                    diversity['image_quality']
                ]
                
                images.append(profile_images)
                ages.append(age)
                diversity_metrics.append(diversity_vector)
                scores.append(score)
                
            except Exception as e:
                print(f"Error loading {profile_folder}: {e}")
                continue
        
        return (np.array(images), np.array(ages), 
                np.array(diversity_metrics), np.array(scores))
    
    def train(self):
        """
        Train the multi-image preference model
        """
        # Load data
        images, ages, diversity_metrics, scores = self.load_training_data()
        
        # Normalize inputs
        ages_scaled = self.scaler.fit_transform(ages.reshape(-1, 1)).flatten()
        diversity_scaled = self.scaler.fit_transform(diversity_metrics)
        
        # Split data
        (X_img_train, X_img_test, X_age_train, X_age_test, 
         X_div_train, X_div_test, y_train, y_test) = train_test_split(
            images, ages_scaled, diversity_scaled, scores, 
            test_size=0.2, random_state=42
        )
        
        # Build models
        self.build_image_model()
        self.build_combined_model()
        
        # Train with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.combined_model.fit(
            [X_img_train, X_age_train, X_div_train],
            y_train,
            validation_data=([X_img_test, X_age_test, X_div_test], y_test),
            epochs=100,
            batch_size=8,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save models
        os.makedirs('models', exist_ok=True)
        self.combined_model.save('models/multi_image_preference_model.h5')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        return history
    
    def predict(self, images, age, diversity_metrics):
        """
        Predict preference score for new profile with multiple images
        """
        if self.combined_model is None:
            self.load_model()
        
        # Preprocess inputs
        images = np.expand_dims(images, axis=0)
        age_scaled = self.scaler.transform([[age]])
        diversity_scaled = self.scaler.transform([diversity_metrics])
        
        # Predict
        prediction = self.combined_model.predict([images, age_scaled, diversity_scaled])
        return float(prediction[0][0])
    
    def load_model(self):
        """Load trained model"""
        try:
            self.combined_model = tf.keras.models.load_model('models/multi_image_preference_model.h5')
            self.scaler = joblib.load('models/scaler.pkl')
        except:
            raise ValueError("No trained model found. Please train first.")