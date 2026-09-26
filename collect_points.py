"""
Batch point-labelling tool — entry point kept for backward compatibility.
Equivalent to ``python -m stereo_gamma collect``.

Put matching images in left/ and right/, then:

    python collect_points.py
"""

from stereo_gamma.collect import main

if __name__ == "__main__":
    main()
