"""Backwards-compatible shim for older src.predict imports and CLI calls.

New code should import from ihsanvision.predict.
"""
if __name__ == "__main__":
    from ihsanvision.predict import main
    main()
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.predict")
