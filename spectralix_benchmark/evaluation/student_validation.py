"""Compatibility facade for student validation module."""

from spectralix_benchmark.evaluation.pipeline.student_validation import *  # noqa: F401,F403


if __name__ == "__main__":
    from spectralix_benchmark.evaluation.cli.student_validation import main

    main()

