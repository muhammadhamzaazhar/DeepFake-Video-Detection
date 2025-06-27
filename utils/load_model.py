import os
from transformers import TimesformerForVideoClassification, TimesformerConfig
from transformers import SwinForImageClassification, SwinConfig
from safetensors.torch import load_file

def load_model(directory: str):
    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        required_files = ["config.json", "model.safetensors"]
        for file in required_files:
            if not os.path.exists(os.path.join(directory, file)):
                raise FileNotFoundError(f"Required file '{file}' is missing in the directory '{directory}'.")

        print(f"Loading model from: {directory}")

        config = TimesformerConfig.from_pretrained(directory)
        model = TimesformerForVideoClassification(config)
        # config = SwinConfig.from_pretrained(directory)
        # model = SwinForImageClassification(config)

        model_dict = load_file(os.path.join(directory, "model.safetensors"))

        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in model_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        excluded_keys = [k for k in model_dict if k not in filtered_state_dict]
        if excluded_keys:
            print(f"Excluded keys due to shape mismatch or absence in model: {excluded_keys}")
        
        model.load_state_dict(filtered_state_dict, strict=False)

        print("Model loaded successfully!")
        # print("Model Architecture\n")
        # print(model)

        return model

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the directory and required files exist.")
        return None

    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None
