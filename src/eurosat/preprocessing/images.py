"""PyTorch dataset of model inputs, with deterministic augmentation views."""
from torch.utils.data import Dataset
from .transforms import image_tensor
from ..cache import key


class Images(Dataset):
    """Model inputs for a list of items.

    items are (image index, view) for one `policy`, or (image index, policy, view) to mix methods in one pass;
    mixed items take their settings from `augmentation` (method -> settings).
    """

    def __init__(self, root, records, items, policy='none', seed=42, params=None, augmentation=None):
        self.root, self.records, self.items = root, records, items
        self.policy, self.seed, self.params, self.augmentation = policy, seed, params, augmentation or {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        if len(item) == 3:
            image_id, policy, view = item
            params = self.augmentation.get(policy)
        else:
            (image_id, view), policy, params = item, self.policy, self.params
        record = self.records[image_id]
        # The seed depends on the image and method only (not the model, baseline or batch order);
        # the view number picks which view, so the views of one image never repeat.
        seed = int(key([self.seed, record['pixels_sha256'], policy])[:15], 16)
        return image_tensor(self.root / record['path'], policy, seed, params, view), record['label']
