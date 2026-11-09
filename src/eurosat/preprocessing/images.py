"""PyTorch image dataset with deterministic support-view seeds."""
from torch.utils.data import Dataset
from .transforms import image_tensor
from ..cache import key


class Images(Dataset):
    def __init__(self, root, records, items, policy, seed, strength=1.):
        self.root, self.records, self.items = root, records, items
        self.policy, self.seed, self.strength = policy, seed, strength

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, view = self.items[index]
        record = self.records[image_id]
        # Image/view seeds are shared across backbones and independent of batch/chunk order.
        seed = int(key([self.seed, record['pixels_sha256'], self.policy, view])[:15], 16)
        return image_tensor(self.root / record['path'], self.policy, seed, self.strength), record['label']
