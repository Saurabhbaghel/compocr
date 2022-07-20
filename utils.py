import os
import glob
from pathlib import Path


def check_dir(dir) -> bool:
    if isinstance(dir,str):
        if os.path.exists(Path(dir))
            return True
        else:
            raise NotADirectoryError(f'{dir} is not a directory.')
    else:
        raise Exception('Enter a valid path.')



