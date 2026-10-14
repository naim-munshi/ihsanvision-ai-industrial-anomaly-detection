"""Backwards-compatible shim for older src.config imports.

New code should import from ihsanvision.config.
"""
if __name__ == "__main__":
    raise SystemExit("Use the canonical ihsanvision.config module instead.")
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.config")
