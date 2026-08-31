"""Shared CLI helpers for the adforce.eval hydra entry points.

The comp/-era argparse flags were replaced by hydra overrides when the module
moved to adforce/eval/. Stale invocations (old docs, shell history, thesis
Makefiles) should fail loudly with the exact translation rather than be
half-parsed by hydra -- so each entry point calls :func:`reject_legacy_flags`
with its flag map before handing argv to hydra. Only known legacy flags are
rejected; hydra's own ``--help``/``--cfg``/etc. pass through untouched.
"""

import sys
from typing import Dict


def reject_legacy_flags(mapping: Dict[str, str], module: str) -> None:
    """Exit with the hydra translation if an old argparse flag is on argv.

    Args:
        mapping: legacy flag -> replacement hydra override (shown verbatim).
        module: dotted module path for the usage line, e.g. "adforce.eval.validate".
    """
    hits = [a for a in sys.argv[1:] if a.split("=")[0] in mapping]
    if not hits:
        return
    lines = [
        f"{module}: argparse flags were replaced by hydra overrides "
        "when comp/ moved to adforce/eval/. Translation:"
    ]
    lines += [f"  {old:<18} ->  {new}" for old, new in mapping.items()]
    lines.append(f"e.g.  python -m {module} " + mapping[hits[0].split("=")[0]])
    raise SystemExit("\n".join(lines))
