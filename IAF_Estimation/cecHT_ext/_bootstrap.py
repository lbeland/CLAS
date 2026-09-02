"""Put the pristine ``cecHT`` submodule on ``sys.path``.

``IAF_Estimation/cecHT`` is a git submodule kept byte-identical to upstream
(``github.com/eikeosmers/cecHT``). All shared code -- ``phase``, ``phase_track``,
``utils`` and the ``EEG/`` helpers -- is imported from there rather than copied.

Import this module (``import _bootstrap``) before importing any of those names.
Every entry point in ``cecHT_ext`` does so after making sure ``cecHT_ext`` itself
is importable, e.g.::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import _bootstrap  # noqa: E402,F401
"""

import sys
from pathlib import Path

_SUBMODULE = Path(__file__).resolve().parent.parent / "cecHT"

if not (_SUBMODULE / "phase.py").exists():
    raise ImportError(
        f"cecHT submodule not found at {_SUBMODULE}. Run "
        "`git submodule update --init IAF_Estimation/cecHT`."
    )

for _p in (_SUBMODULE, _SUBMODULE / "EEG"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
