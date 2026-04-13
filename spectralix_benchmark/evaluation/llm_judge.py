"""Compatibility facade for judge module."""

from spectralix_benchmark.evaluation.pipeline.llm_judge import *  # noqa: F401,F403


if __name__ == "__main__":
    from spectralix_benchmark.evaluation.cli.llm_judge import main

    main()

