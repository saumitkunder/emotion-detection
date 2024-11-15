import h5py

# Path to the model file
MODEL_PATH = "app/model/ann_model.h5"

def fix_model_config(model_path):
    """
    Fix the model configuration to remove unsupported 'batch_shape' argument.
    Args:
        model_path: Path to the H5 model file.
    """
    try:
        with h5py.File(model_path, "r+") as f:
            # Load the model configuration
            config = f.attrs["model_config"]

            # Ensure `config` is a string
            if isinstance(config, bytes):
                config = config.decode("utf-8")

            # Remove 'batch_shape' argument
            fixed_config = config.replace("'batch_shape': [None, 60],", "")

            # Save the updated configuration
            f.attrs["model_config"] = fixed_config
            print(f"Successfully fixed the model configuration in {model_path}")

    except Exception as e:
        print(f"Error fixing the model configuration: {e}")

# Run the fix
fix_model_config(MODEL_PATH)
