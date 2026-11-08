from pathlib import Path
from redteam.shortcuts import main
main('shortcut_copy_input', Path('redteam/.cache/usable_reference'), Path('artifacts'))
