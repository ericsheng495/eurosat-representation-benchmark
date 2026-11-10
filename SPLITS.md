# Shared base/novel splits

Use the committed `splits_manifest.json`; do not regenerate it separately for each teammate. It was generated once with the shared `make_splits.py` (only line endings normalized for Git; split logic unchanged) against the complete, checksum-verified EuroSAT RGB archive.

- Seed: 42.
- Six base classes: 10,200 training, 3,400 validation and 3,400 test images (60/20/20).
- Four novel classes: 3,000 development-pool and 7,000 test-pool images (30/70).
- Per shot setting (1, 5, 10): 100 development episodes and 600 test episodes.
- Each episode is four-way, with k support images and 15 query images per class. Support/query are disjoint within each episode; development and test pools are disjoint. Images can recur across episodes, and supports are not guaranteed nested across shot settings.

Paths in the manifest are filenames relative to their class folders. Each person supplies their own image root containing all ten class folders:

```python
from make_splits import load_manifest, base_items, episode_items, n_episodes

manifest = load_manifest('splits_manifest.json')
IMG_ROOT = '/content/EuroSAT/2750'  # Adjust to your own extracted folder.
train = base_items(manifest, IMG_ROOT, 'train')
support, query = episode_items(manifest, IMG_ROOT, 'test', 5, 0)
print(n_episodes(manifest, 'test', 5))  # 600
```

`base_items` labels the six base classes from 0 to 5; `episode_items` labels the four novel classes from 0 to 3. Use the class order in `manifest['meta']` when interpreting labels. Development episodes are for choosing settings; use fixed test episodes only after settings are locked.

The main notebook uses this fixed four-class protocol exclusively. Six base classes are reserved, and no backbone is trained on their partitions. Set `MANIFEST_FILE` or `EUROSAT_SPLITS` to the committed file, or copy it into shared storage. Every pool and episode is validated before selecting a smoke/full subset; selected records and assignments are exported for inspection.

Generation command used from the repository root:

```sh
python make_splits.py data/extracted/8ebea626349354c5/2750 splits_manifest.json
```

The ZIP, image files, weights and embeddings remain outside Git; only the split definitions and loader script are shared here.
