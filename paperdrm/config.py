"""
Compatibility shim: config classes/functions now live in paperdrm.settings.

TODO: remove in a future release after callers switch to `paperdrm.settings`.
"""

import warnings

warnings.warn(
    "paperdrm.config is deprecated; import from paperdrm.settings instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .settings import *  # noqa: F401,F403
