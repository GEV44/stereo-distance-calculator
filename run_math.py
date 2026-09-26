"""
Interactive stereo distance measurement — entry point kept for backward
compatibility.  Equivalent to ``python -m stereo_gamma measure LEFT RIGHT``.

    python run_math.py left.jpg right.jpg

The implementation lives in the ``stereo_gamma`` package (see README).
"""

import sys

from stereo_gamma.app import run

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python run_math.py <left_image> <right_image>")
    run(sys.argv[1], sys.argv[2])
