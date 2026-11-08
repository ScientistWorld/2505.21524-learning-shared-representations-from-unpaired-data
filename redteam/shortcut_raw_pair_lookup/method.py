from pathlib import Path
from redteam.shortcuts import main
main('shortcut_raw_pair_lookup', Path('redteam/.cache/original_artifacts'), Path('artifacts'))
