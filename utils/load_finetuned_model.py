import torch
from transformers import TimesformerForVideoClassification, TimesformerConfig

def load_finetuned_model(weights_path: str, config_path: str, device):
    try:
        config = TimesformerConfig.from_pretrained(config_path)
        model = TimesformerForVideoClassification(config)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        print("Fine-tuned model loaded successfully for evaluation!")
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.numel()} trainable={param.requires_grad}")
        return model

    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None
