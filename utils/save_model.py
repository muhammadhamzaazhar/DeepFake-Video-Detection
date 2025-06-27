from transformers import TimesformerForVideoClassification
from transformers import SwinForImageClassification

def save_pretrained_model(model_name: str, save_directory: str, swin_b=False) -> None:
    try:
        print(f"Loading model: {model_name}")
        if swin_b:
            model = SwinForImageClassification.from_pretrained(model_name)
        else:
            model = TimesformerForVideoClassification.from_pretrained(model_name)

        print(f"Saving model to: {save_directory}")
        model.save_pretrained(save_directory)
        print("Model saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")