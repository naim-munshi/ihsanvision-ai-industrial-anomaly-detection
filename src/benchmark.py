"""Backwards-compatible shim for older src.benchmark imports and CLI calls.

New code should import from ihsanvision.benchmark.
"""
if __name__ == "__main__":
    from ihsanvision.benchmark import main
    main()
else:
    from importlib import import_module
    import sys
    sys.modules[__name__] = import_module("ihsanvision.benchmark")
