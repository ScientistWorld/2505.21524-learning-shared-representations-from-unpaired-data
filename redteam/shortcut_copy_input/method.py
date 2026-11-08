from pathlib import Path
from redteam.shortcuts import main
main('shortcut_copy_input', Path('redteam/.cache/original_artifacts'), Path('artifacts'))
