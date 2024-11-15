import tensorflow as tf

# Path to the existing model
OLD_MODEL_PATH = "/Users/saumit/Projects/emotion-detection/app/model/ann_model.h5ann_model.h5"
NEW_MODEL_PATH = "/Users/saumit/Projects/emotion-detection/app/model/ann_model.h5/fixed_audio_model.h5"

def resave_model(old_model_path, new_model_path):
    """
    Load an existing model and re-save it in a newer format to remove compatibility issues.
    """
    try:
        # Load the old model
        model = tf.keras.models.load_model(old_model_path)
        print("Model loaded successfully. Re-saving the model...")

        # Save the model in the new format
        model.save(new_model_path, save_format="h5")
        print(f"Model successfully re-saved to: {new_model_path}")
    except Exception as e:
        print(f"Error re-saving the model: {e}")

# Re-save the model
resave_model(OLD_MODEL_PATH, NEW_MODEL_PATH)
