# Shared splits

Everyone uses the committed `splits_manifest.json`. Do not make your own. It was generated once by `make_splits.py` (seed 42) from the checksum-verified EuroSAT RGB archive.

| Part | Classes | Images |
|---|---|---|
| Fine-tuning | AnnualCrop, Forest, Highway, Industrial, Residential, SeaLake | 10,200 train · 3,400 val · 3,400 test (60/20/20) |
| Few-shot evaluation | HerbaceousVegetation, Pasture, PermanentCrop, River | 3,000 development · 7,000 test (30/70) |

- For each shot budget (1, 5, 10): 100 development episodes and 600 test episodes.
- An episode is 4-way: k support images and 15 query images per class. Supports and queries never overlap.
- Development and test pools never overlap. Images can recur across episodes. Supports are not nested across shot budgets.
- The six fine-tuning classes train only the fine-tuned baseline. The four evaluated classes never train any backbone.

The notebook checks every pool and episode before it selects the episodes for a run. To use another copy of the manifest, set `MANIFEST_FILE` in the notebook (or `EUROSAT_SPLITS`).

## Using the manifest outside the notebook

Paths are filenames inside each class folder. Point `IMG_ROOT` at your own folder that holds all ten class folders:

```python
from make_splits import load_manifest, base_items, episode_items, n_episodes

manifest = load_manifest('splits_manifest.json')
IMG_ROOT = '/content/EuroSAT/2750'  # your extracted folder
train = base_items(manifest, IMG_ROOT, 'train')
support, query = episode_items(manifest, IMG_ROOT, 'test', 5, 0)
print(n_episodes(manifest, 'test', 5))  # 600
```

`base_items` labels the six fine-tuning classes 0–5. `episode_items` labels the four evaluated classes 0–3. The class order is in `manifest['meta']`.

The manifest was made with:

```sh
python make_splits.py data/extracted/8ebea626349354c5/2750 splits_manifest.json
```

Git stores only the manifest and `make_splits.py`. The archive, images, weights and embeddings stay outside Git.
