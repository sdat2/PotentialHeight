"""
Utility re-exports for the cle15 package.

Re-exports :func:`w22.utils.pressure_from_wind` so that the moved
cle15 modules can use ``from .utils import pressure_from_wind`` without change.
"""

from w22.utils import pressure_from_wind  # noqa: F401
