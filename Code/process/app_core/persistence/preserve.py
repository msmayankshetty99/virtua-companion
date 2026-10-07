"""Keep the bytes of an unreadable user file before anything replaces it."""
from datetime import datetime
import shutil


def preserve_unreadable(path):
    """Copy path to <name>.unreadable-<timestamp>, or move it there when it cannot be read.

    Moving needs only directory access, so a permission-denied file can still be kept.
    Returns the backup path, or None if the file no longer exists. Raises OSError when
    the file can be neither copied nor moved; callers must then not overwrite it.
    """
    backup = path.with_name(f'{path.name}.unreadable-{datetime.now():%Y%m%d-%H%M%S-%f}')
    try: shutil.copy2(path, backup)
    except FileNotFoundError: return None
    except OSError:
        try: path.replace(backup)
        except FileNotFoundError: return None
    return backup
