# mergekit-cabs
# SPDX-License-Identifier: MIT
"""mergekit-cabs: CABS and CABS+ merge methods for mergekit.

Importing this package registers the ``cabs`` and ``cabs_plus`` merge
methods into mergekit's registry so they can be used directly in YAML
recipes::

    import mergekit_cabs  # registers cabs / cabs_plus
    from mergekit.scripts.run_yaml import main
    main()

or through the bundled ``mergekit-cabs-yaml`` console script / CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["register", "registered_variants", "CABS", "CABS_PLUS"]

if TYPE_CHECKING:  # pragma: no cover
    from .methods import CABS, CABS_PLUS


def _import_methods():
    from .methods import CABS, CABS_PLUS

    return CABS, CABS_PLUS


def register(force: bool = False) -> None:
    """Register ``cabs`` and ``cabs_plus`` into mergekit's method registry.

    Idempotent by default; pass ``force=True`` to overwrite existing
    registrations (useful when reloading a modified implementation).
    """
    from mergekit.merge_methods import registry

    cabs, cabs_plus = _import_methods()
    for method in (cabs, cabs_plus):
        exists = True
        try:
            registry.get(method.spec.name)
        except RuntimeError:
            exists = False
        if exists and force:
            registry._METHODS.pop(method.spec.name, None)
            exists = False
        if not exists:
            registry.register(method)


def registered_variants():
    """Return the subset of {cabs, cabs_plus} currently registered."""
    from mergekit.merge_methods import registry

    out = []
    for name in ("cabs", "cabs_plus"):
        try:
            registry.get(name)
            out.append(name)
        except RuntimeError:
            pass
    return tuple(out)


register()
