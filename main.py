from __future__ import annotations

import sys

from dinoia.__main__ import main as module_main


if __name__ == "__main__":
    raise SystemExit(module_main(sys.argv[1:]))
