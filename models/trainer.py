from combined_model import MultiImagePreferenceModel
import os

def train_model():
    """
    Train the preference learning model
    """
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize and train model
    model = MultiImagePreferenceModel()
    
    try:
        history = model.train()
        print("Model training completed successfully!")
        
        # Print training results
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        print(f"Final Loss: {final_loss:.4f}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        
    except ValueError as e:
        print(f"Training failed: {e}")
    except Exception as e:
        print(f"Unexpected error during training: {e}")

if __name__ == "__main__":
    train_model()