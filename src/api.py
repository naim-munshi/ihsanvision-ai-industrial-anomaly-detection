"""Backwards-compatible shim for older src.api imports.

New code should import from ihsanvision.api.
"""
if __name__ == "__main__":
    raise SystemExit("Use the canonical ihsanvision.api module instead.")
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.api")
