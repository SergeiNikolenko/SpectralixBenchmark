"""Compatibility facade for full matrix module."""

from spectralix_benchmark.evaluation.pipeline.run_full_matrix import *  # noqa: F401,F403


if __name__ == "__main__":
    from spectralix_benchmark.evaluation.cli.run_full_matrix import main

    main()

