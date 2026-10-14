"""Backwards-compatible shim for older src.train imports and CLI calls.

New code should import from ihsanvision.train.
"""
if __name__ == "__main__":
    from ihsanvision.train import main
    main()
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.train")
