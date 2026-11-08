from pathlib import Path
from redteam.shortcuts import main
main('shortcut_raw_pair_lookup', Path('redteam/.cache/usable_reference'), Path('artifacts'))
