"""Provenance for every data file in this gym (standard library only).

Rebuilds, from the public UCI Multiple Features archive, the three disjoint
parts of the Handwritten weakly paired retrieval task:

* public training data  -> data/handwritten/
    100 known cross-view pairs plus 990 unpaired rows per view (the remaining
    1100 training rows, 10% dropped independently per view, then shuffled
    independently per view -- Appendix E.7 of the paper);
* public development pairs -> eval/train/data/dev_{kar,pix}.txt (400 pairs);
* hidden held-out test pairs -> eval/test/data/heldout_{kar,pix}.txt (400 pairs).

Test and development rows are stratified (40 per digit class) and stored in
a random order.  The split seed below exists only here, so the identity of
the held-out rows cannot be regenerated from anything outside eval/test/.
Rows are copied verbatim from the archive text files.

Usage (needs internet only if --archive is not given):
    python3 eval/test/build_split.py --check            # verify committed files
    python3 eval/test/build_split.py --write            # (re)write all files
    python3 eval/test/build_split.py --archive PATH ... # use a local archive copy
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UCI_URL = "https://archive.ics.uci.edu/static/public/72/multiple+features.zip"
UCI_SHA256 = "898a50a7637f1ed5a8cee2493aaa0e7f4d52795c412f59b0969ec7c3046ee7bd"
SPLIT_SEED = 3776266133972789635
N_PER_CLASS = 200
N_CLASSES = 10
TEST_PER_CLASS = 40
DEV_PER_CLASS = 40
N_PAIRED = 100
DROP_FRACTION = 0.10
MANIFEST = ROOT / "eval" / "test" / "data" / "split_manifest.json"


def read_archive(path: Path) -> dict[str, list[str]]:
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != UCI_SHA256:
        raise SystemExit(f"archive checksum mismatch: {digest}")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        views = {
            "kar": archive.read("mfeat-kar").decode("ascii").splitlines(),
            "pix": archive.read("mfeat-pix").decode("ascii").splitlines(),
        }
    for name, lines in views.items():
        if len(lines) != N_PER_CLASS * N_CLASSES:
            raise SystemExit(f"unexpected row count in mfeat-{name}: {len(lines)}")
    return views


def make_split() -> dict[str, list[int]]:
    rng = random.Random(SPLIT_SEED)
    test: list[int] = []
    dev: list[int] = []
    train: list[int] = []
    for label in range(N_CLASSES):
        rows = list(range(label * N_PER_CLASS, (label + 1) * N_PER_CLASS))
        rng.shuffle(rows)
        test += rows[:TEST_PER_CLASS]
        dev += rows[TEST_PER_CLASS:TEST_PER_CLASS + DEV_PER_CLASS]
        train += rows[TEST_PER_CLASS + DEV_PER_CLASS:]
    rng.shuffle(test)
    rng.shuffle(dev)
    rng.shuffle(train)
    paired = train[:N_PAIRED]
    pool = train[N_PAIRED:]
    keep = int(round((1.0 - DROP_FRACTION) * len(pool)))
    unpaired_kar = rng.sample(pool, keep)
    unpaired_pix = rng.sample(pool, keep)
    return {
        "test": test,
        "dev": dev,
        "paired": paired,
        "unpaired_kar": unpaired_kar,
        "unpaired_pix": unpaired_pix,
    }


def outputs(split: dict[str, list[int]]) -> dict[Path, tuple[str, list[int]]]:
    data = ROOT / "data" / "handwritten"
    dev = ROOT / "eval" / "train" / "data"
    test = ROOT / "eval" / "test" / "data"
    return {
        data / "paired_kar.txt": ("kar", split["paired"]),
        data / "paired_pix.txt": ("pix", split["paired"]),
        data / "unpaired_kar.txt": ("kar", split["unpaired_kar"]),
        data / "unpaired_pix.txt": ("pix", split["unpaired_pix"]),
        dev / "dev_kar.txt": ("kar", split["dev"]),
        dev / "dev_pix.txt": ("pix", split["dev"]),
        test / "heldout_kar.txt": ("kar", split["test"]),
        test / "heldout_pix.txt": ("pix", split["test"]),
    }


def render(lines: list[str], rows: list[int]) -> str:
    return "".join(lines[i].rstrip() + "\n" for i in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.archive is None:
        tmp = Path(tempfile.mkdtemp())
        args.archive = tmp / "multiple_features.zip"
        urllib.request.urlretrieve(UCI_URL, args.archive)
    views = read_archive(args.archive)
    split = make_split()

    all_rows = split["test"] + split["dev"] + split["paired"] + split["unpaired_kar"]
    if len(set(split["test"]) | set(split["dev"]) | set(split["paired"])) != 900:
        raise SystemExit("split parts overlap")
    if set(split["unpaired_kar"]) & (set(split["test"]) | set(split["dev"]) | set(split["paired"])):
        raise SystemExit("unpaired pool overlaps held-out or paired rows")
    del all_rows

    manifest = {
        "source": UCI_URL,
        "source_sha256": UCI_SHA256,
        "split_seed": SPLIT_SEED,
        "rows": split,
        "labels": {name: [i // N_PER_CLASS for i in rows] for name, rows in split.items()},
        "files": {},
    }
    failures = []
    for path, (view, rows) in outputs(split).items():
        text = render(views[view], rows)
        rel = str(path.relative_to(ROOT))
        manifest["files"][rel] = hashlib.sha256(text.encode("ascii")).hexdigest()
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        elif not path.exists() or path.read_text() != text:
            failures.append(rel)
    if args.write:
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(manifest['files'])} data files and {MANIFEST.relative_to(ROOT)}")
    elif failures:
        raise SystemExit("files differ from the archive-derived split: " + ", ".join(failures))
    else:
        print("all committed data files match the public archive")


if __name__ == "__main__":
    main()
