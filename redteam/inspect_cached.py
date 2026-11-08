"""Light audit diagnostics from saved candidates, using only public train rules."""
import json
from pathlib import Path

import numpy as np
from eval.train.evaluate import stratified_slice_indices


for cache in sorted(Path('redteam/.cache/cases').iterdir()):
    summaries = {}
    for path in cache.glob('*.npz'):
        with np.load(path, allow_pickle=False) as data:
            labels = data['labels']
            try:
                train_queries = len(stratified_slice_indices(labels))
            except ValueError:
                train_queries = 0
            summaries[path.name] = dict(
                gallery_rows=int(data['x_embeddings'].shape[0]),
                embedding_dimensions=int(data['x_embeddings'].shape[1]),
                label_classes=int(len(np.unique(labels))),
                visible_train_queries=train_queries,
                complementary_test_queries=int(len(labels) - train_queries),
            )
    folder = Path('redteam') / cache.name
    (folder / 'artifact_summary.json').write_text(json.dumps(summaries, indent=2, sort_keys=True))
