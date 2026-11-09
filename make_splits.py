"""EuroSAT split manifest: generate once, then everyone loads the same file.

ONE PERSON, ONCE (on the extracted EuroSAT folder; a local Colab copy is safest):
    python make_splits.py /content/EuroSAT /content/drive/MyDrive/.../splits_manifest.json

EVERYONE (in their own notebook/scripts):
    from make_splits import load_manifest, base_items, episode_items, n_episodes
    m = load_manifest('splits_manifest.json')
    train = base_items(m, IMG_ROOT, 'train')            # [(path, label), ...]  6 base classes
    sup, qry = episode_items(m, IMG_ROOT, 'test', 5, 0)  # 5-shot test episode #0, 4 novel classes
IMG_ROOT = folder that contains the 10 class folders (can differ per person).
"""
import os, json, random, hashlib

# ---------------- CONFIG ----------------
SEED = 42
BASE_CLASSES  = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'SeaLake']
NOVEL_CLASSES = ['HerbaceousVegetation', 'Pasture', 'PermanentCrop', 'River']
TRAIN_FRAC, VAL_FRAC = 0.60, 0.2            # test = remainder
NOVEL_DEV_FRAC = 0.30                        # novel images reserved for dev episodes (tuning)
N_QUERY = 15                                 # query images per class per episode
SHOTS = [1, 5, 10]
N_DEV_EPISODES, N_TEST_EPISODES = 100, 600   # per shot setting

# Official EuroSAT RGB image counts. Used to catch Google Drive's incomplete listings.
EXPECTED_COUNTS = {
    'AnnualCrop': 3000, 'Forest': 3000, 'HerbaceousVegetation': 3000, 'Highway': 2500,
    'Industrial': 2500, 'Pasture': 2000, 'PermanentCrop': 2500, 'Residential': 3000,
    'River': 2500, 'SeaLake': 3000,
}
# -----------------------------------------

def rng_for(*parts):
    """Deterministic RNG independent of listing order, machine, and Python hash seed."""
    key = "|".join(str(p) for p in (SEED, *parts))
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))

def list_images(source_dir, cls):
    folder = os.path.join(source_dir, cls)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing class folder: {folder}")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')))
    if len(files) != EXPECTED_COUNTS[cls]:
        raise RuntimeError(
            f"{cls}: found {len(files)} images, expected {EXPECTED_COUNTS[cls]}. "
            "Drive listing is probably incomplete -- re-mount Drive (force_remount=True) or "
            "extract the zip to local /content and retry.")
    return files

def build_manifest(source_dir):
    assert not set(BASE_CLASSES) & set(NOVEL_CLASSES), "base/novel overlap"
    assert set(BASE_CLASSES) | set(NOVEL_CLASSES) == set(EXPECTED_COUNTS), "class lists don't cover all 10 classes"

    m = {'meta': {'seed': SEED, 'base_classes': BASE_CLASSES, 'novel_classes': NOVEL_CLASSES,
                  'n_query': N_QUERY, 'shots': SHOTS},
         'base': {'train': {}, 'val': {}, 'test': {}},
         'novel': {'dev_pool': {}, 'test_pool': {}},
         'episodes': {'dev': {}, 'test': {}}}

    # Base classes: deterministic stratified 60/20/20
    for cls in BASE_CLASSES:
        files = list_images(source_dir, cls)
        rng_for('base', cls).shuffle(files)
        n_tr, n_va = int(len(files) * TRAIN_FRAC), int(len(files) * VAL_FRAC)
        m['base']['train'][cls] = sorted(files[:n_tr])
        m['base']['val'][cls]   = sorted(files[n_tr:n_tr + n_va])
        m['base']['test'][cls]  = sorted(files[n_tr + n_va:])

    # Novel classes: disjoint dev/test pools so tuning never touches final-eval images
    for cls in NOVEL_CLASSES:
        files = list_images(source_dir, cls)
        rng_for('novel', cls).shuffle(files)
        n_dev = int(len(files) * NOVEL_DEV_FRAC)
        m['novel']['dev_pool'][cls]  = sorted(files[:n_dev])
        m['novel']['test_pool'][cls] = sorted(files[n_dev:])

    # Fixed few-shot episodes (4-way, K-shot). Everyone evaluates on the same ones.
    for split, pool, n_eps in [('dev', 'dev_pool', N_DEV_EPISODES), ('test', 'test_pool', N_TEST_EPISODES)]:
        for k in SHOTS:
            eps = []
            for e in range(n_eps):
                ep = {}
                for cls in NOVEL_CLASSES:
                    picked = rng_for('episode', split, k, e, cls).sample(m['novel'][pool][cls], k + N_QUERY)
                    ep[cls] = {'support': picked[:k], 'query': picked[k:]}
                eps.append(ep)
            m['episodes'][split][f'{k}shot'] = eps
    return m

def verify(m, source_dir):
    # base: no leakage, full coverage
    for cls in BASE_CLASSES:
        tr, va, te = (set(m['base'][s][cls]) for s in ('train', 'val', 'test'))
        assert not (tr & va or tr & te or va & te), f"{cls}: base split overlap"
        assert len(tr) + len(va) + len(te) == EXPECTED_COUNTS[cls], f"{cls}: base split lost images"
    # novel pools disjoint
    for cls in NOVEL_CLASSES:
        d, t = set(m['novel']['dev_pool'][cls]), set(m['novel']['test_pool'][cls])
        assert not d & t and len(d) + len(t) == EXPECTED_COUNTS[cls], f"{cls}: novel pool problem"
    # episodes: support/query disjoint, drawn from correct pool, right sizes
    for split, pool in [('dev', 'dev_pool'), ('test', 'test_pool')]:
        for key, eps in m['episodes'][split].items():
            k = int(key.replace('shot', ''))
            for ep in eps:
                assert set(ep) == set(NOVEL_CLASSES)
                for cls, d in ep.items():
                    s, q = d['support'], d['query']
                    assert len(s) == k and len(q) == N_QUERY and not set(s) & set(q)
                    assert set(s + q) <= set(m['novel'][pool][cls]), "episode image outside its pool"
    # every referenced file actually exists on disk
    for cls in BASE_CLASSES + NOVEL_CLASSES:
        assert os.path.isdir(os.path.join(source_dir, cls))
    print("All checks passed.")
    for s in ('train', 'val', 'test'):
        print(f"  base/{s}: {sum(len(v) for v in m['base'][s].values())} images")
    for p in ('dev_pool', 'test_pool'):
        print(f"  novel/{p}: {sum(len(v) for v in m['novel'][p].values())} images")


# ---------------- LOADER (what teammates use) ----------------
def load_manifest(path):
    with open(path) as f:
        return json.load(f)

def base_items(m, img_root, split):
    """split in {'train','val','test'} -> [(path, label_idx)] over the 6 base classes."""
    classes = m['meta']['base_classes']
    return [(os.path.join(img_root, c, f), i)
            for i, c in enumerate(classes) for f in m['base'][split][c]]

def episode_items(m, img_root, split, shots, idx):
    """split in {'dev','test'}, shots in {1,5,10} -> (support, query) lists of (path, label_idx), 4 novel classes."""
    classes = m['meta']['novel_classes']
    ep = m['episodes'][split][f'{shots}shot'][idx]
    sup = [(os.path.join(img_root, c, f), i) for i, c in enumerate(classes) for f in ep[c]['support']]
    qry = [(os.path.join(img_root, c, f), i) for i, c in enumerate(classes) for f in ep[c]['query']]
    return sup, qry

def n_episodes(m, split, shots):
    return len(m['episodes'][split][f'{shots}shot'])

if __name__ == '__main__':
    import sys
    source_dir, out_path = sys.argv[1], sys.argv[2]
    m = build_manifest(source_dir)
    verify(m, source_dir)
    with open(out_path, 'w') as f:
        json.dump(m, f)
    print("Wrote", out_path, f"({os.path.getsize(out_path)/1e6:.1f} MB)")