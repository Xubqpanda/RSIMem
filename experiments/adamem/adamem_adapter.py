"""Experiment-facing facade for the host-neutral AdaMem policy contract."""

import sys as _sys

from rsimem.memory import adamem_adapter as _implementation

_sys.modules[__name__] = _implementation
