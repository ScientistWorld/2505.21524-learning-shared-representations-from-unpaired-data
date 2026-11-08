from pathlib import Path
from redteam.shortcuts import main
main('shortcut_replace_baseline', Path('redteam/.cache/original_artifacts'), Path('artifacts'))
