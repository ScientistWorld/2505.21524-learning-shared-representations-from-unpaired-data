from pathlib import Path
from redteam.shortcuts import main
main('shortcut_replace_baseline', Path('redteam/.cache/usable_reference'), Path('artifacts'))
