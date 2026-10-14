"""Backwards-compatible shim for older src.schemas imports.

New code should import from ihsanvision.schemas.
"""
if __name__ == "__main__":
    raise SystemExit("Use the canonical ihsanvision.schemas module instead.")
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.schemas")
