"""Backwards-compatible shim for older src.serving imports.

New code should import from ihsanvision.serving.
"""
if __name__ == "__main__":
    raise SystemExit("Use the canonical ihsanvision.serving module instead.")
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.serving")
