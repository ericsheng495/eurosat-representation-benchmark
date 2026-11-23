"""Training code. The only training in this project is fine-tuning for baseline 3."""
from .finetune import RECIPE, fine_tune, finetuned_weights_dir

__all__ = ['RECIPE', 'fine_tune', 'finetuned_weights_dir']
