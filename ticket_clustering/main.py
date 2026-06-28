"""Project-local CLI wrapper.

This file lets the assignment structure run from either the repository root or
inside the ``ticket_clustering`` folder while keeping one implementation in the
root ``main.py`` entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from main import main  # noqa: E402


if __name__ == "__main__":
    main()

