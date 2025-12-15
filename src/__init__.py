"""
Compatibility shim for legacy imports.

The package has been renamed to ``paperdrm``. Imports through ``src`` will
continue to work for now but will emit a deprecation warning.
"""

from __future__ import annotations

import sys
import warnings

from paperdrm import *  # noqa: F401,F403
from paperdrm import (
    config,
    drp_analysis,
    drp_compute,
    drp_direction,
    drp_plot,
    image_io,
    imagepack,
    imageparam,
    line_detection,
    paths,
    roi,
)

warnings.warn(
    "Imports from 'src' are deprecated; use 'paperdrm' instead.",
    DeprecationWarning,
    stacklevel=2,
)

_ALIASES = {
    "config": config,
    "drp_analysis": drp_analysis,
    "drp_compute": drp_compute,
    "drp_direction": drp_direction,
    "drp_plot": drp_plot,
    "image_io": image_io,
    "imagepack": imagepack,
    "imageparam": imageparam,
    "line_detection": line_detection,
    "paths": paths,
    "roi": roi,
}

for name, module in _ALIASES.items():
    sys.modules[f"{__name__}.{name}"] = module
