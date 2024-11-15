# import tensorflow as tf

# # Path to the existing model
# OLD_MODEL_PATH = "/Users/saumit/Projects/emotion-detection/app/model/ann_model.h5"
# NEW_MODEL_PATH = "/Users/saumit/Projects/emotion-detection/app/model/ann_model.h5/fixed_audio_model.h5"


# def rebuild_model(old_model_path, new_model_path):
#     """
#     Rebuild the model to fix configuration issues and re-save it.
#     """
#     try:
#         # Load the existing model
#         old_model = tf.keras.models.load_model(old_model_path)
#         print("Model loaded successfully. Rebuilding...")

#         # Reconstruct the model
#         # Extract the original configuration
#         config = old_model.get_config()

#         # Fix any unsupported configurations
#         for layer in config['layers']:
#             if 'batch_shape' in layer['config']:
#                 del layer['config']['batch_shape']

#         # Rebuild the model using the modified configuration
#         new_model = tf.keras.models.Model.from_config(config)

#         # Copy the weights from the old model
#         new_model.set_weights(old_model.get_weights())

#         # Save the new model
#         new_model.save(new_model_path, save_format="h5")
#         print(f"Model successfully re-saved to: {new_model_path}")
#     except Exception as e:
#         print(f"Error re-saving the model: {e}")

# # Run the rebuild
# rebuild_model(OLD_MODEL_PATH, NEW_MODEL_PATH)