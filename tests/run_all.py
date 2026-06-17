#!/usr/bin/env python3
"""Run all MIST-Sleep unit tests + the smoke test (pytest-free, CPU, fast).

    python tests/run_all.py

Each test module exposes test_* functions and an importable runner; we discover
and call every test_* function and report pass/fail counts.
"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_MODULES = [
    "tests.test_synthetic",
    "tests.test_band_metrics",
    "tests.test_losses",
    "tests.test_probes",
    "tests.test_normalization",
    "tests.test_metrics",
    "tests.test_trainer_ablations",
]


def main():
    passed, failed = 0, 0
    for modname in TEST_MODULES:
        mod = importlib.import_module(modname)
        fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")
               and callable(getattr(mod, n))]
        for fn in fns:
            try:
                fn()
                passed += 1
                print(f"PASS {modname}.{fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {modname}.{fn.__name__}")
                traceback.print_exc()

    # Smoke test (separate top-level script)
    print("\n--- end-to-end smoke test ---")
    try:
        smoke = importlib.import_module("tests.smoke_test")
        smoke.main()
        passed += 1
    except Exception:
        failed += 1
        traceback.print_exc()

    print(f"\n==== {passed} passed, {failed} failed ====")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
