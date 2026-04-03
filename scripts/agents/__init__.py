"""Agent runtime package for SpectralixBenchmark."""

__all__ = ["AgentRuntime", "AgentRuntimeError"]


def __getattr__(name: str):
    if name in {"AgentRuntime", "AgentRuntimeError"}:
        from .runtime import AgentRuntime, AgentRuntimeError

        exports = {
            "AgentRuntime": AgentRuntime,
            "AgentRuntimeError": AgentRuntimeError,
        }
        return exports[name]
    raise AttributeError(name)
