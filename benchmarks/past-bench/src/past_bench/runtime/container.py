"""Docker lifecycle management for agent runtime containers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import RuntimeConfig
from .registry import AgentSpec, env_name_candidates, resolve_env_value


@dataclass
class RuntimeContainerHandle:
    """Reference to a running runtime container."""

    container: Any
    host_port: int
    run_id: str
    base_url: str


class RuntimeContainerManager:
    """Manages the generic agent runtime container."""

    _HERMES_PAST_BENCH_TOOLS_PROBE = (
        "from past_bench.runtime.adapters.hermes import HermesAdapter; "
        "import sys; "
        "sys.exit(0 if hasattr(HermesAdapter, '_register_past_bench_tools') else 17)"
    )

    def __init__(self, runtime_config: RuntimeConfig, *, image: str | None = None) -> None:
        try:
            import docker  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "docker package is required for runtime container mode. "
                "Install with: pip install 'past-bench[sandbox]'"
            ) from None

        self._config = runtime_config
        self._image = image or runtime_config.image
        kwargs: dict[str, Any] = {}
        if runtime_config.docker_host:
            kwargs["base_url"] = runtime_config.docker_host
        self._docker = docker.from_env(**kwargs)

    @staticmethod
    def _proxy_env() -> dict[str, str]:
        env = {}
        for key in (
            "http_proxy", "https_proxy", "no_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    def start_container(
        self,
        *,
        run_id: str,
        agent_spec: AgentSpec,
        registry_path: str | None = None,
    ) -> RuntimeContainerHandle:
        env = self._proxy_env()
        for key in agent_spec.required_env:
            value = resolve_env_value(key)
            if value:
                for candidate in env_name_candidates(key):
                    env[candidate] = value

        cache_dir = Path(self._config.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        volumes: dict[str, dict[str, str]] = {
            str(cache_dir): {"bind": "/opt/runtime-cache", "mode": "rw"},
        }
        env["PAST_BENCH_RUNTIME_CACHE"] = "/opt/runtime-cache"

        if registry_path:
            registry_abs = Path(registry_path).expanduser().resolve()
            if registry_abs.exists():
                volumes[str(registry_abs)] = {"bind": "/opt/past-bench/agents.yaml", "mode": "ro"}
                env["PAST_BENCH_AGENT_REGISTRY"] = "/opt/past-bench/agents.yaml"

        container = self._docker.containers.run(
            image=self._image,
            detach=True,
            name=f"past-bench-runtime-{run_id}",
            ports={f"{self._config.server_port}/tcp": None},
            labels={"app": "past-bench", "role": "runtime", "run_id": run_id},
            environment=env,
            volumes=volumes,
            extra_hosts={self._config.host_alias: "host-gateway"},
        )
        try:
            host_port = self._get_mapped_port(container)
            base_url = f"http://localhost:{host_port}"
            self._wait_healthy(f"{base_url}/health")
            self._verify_agent_runtime(container, agent_spec)
        except Exception:
            try:
                container.remove(force=True)
            except Exception:
                pass
            raise
        print(f"[runtime] Container past-bench-runtime-{run_id} started at {base_url}")
        return RuntimeContainerHandle(
            container=container,
            host_port=host_port,
            run_id=run_id,
            base_url=base_url,
        )

    def stop_container(self, handle: RuntimeContainerHandle) -> None:
        try:
            handle.container.remove(force=True)
            print(f"[runtime] Container past-bench-runtime-{handle.run_id} removed")
        except Exception as exc:
            print(f"[runtime] Warning: failed to remove runtime container: {exc}")

    def build_image(self, context_path: str = ".", *, dockerfile: str = "Dockerfile.runtime") -> str:
        context_abs = str(Path(context_path).resolve())
        print(f"[runtime] Building image {self._image} from {context_abs} (dockerfile={dockerfile}) ...")
        image, logs = self._docker.images.build(
            path=context_abs,
            dockerfile=dockerfile,
            tag=self._image,
            rm=True,
        )
        for chunk in logs:
            if "stream" in chunk:
                line = chunk["stream"].rstrip()
                if line:
                    print(f"  {line}")
        print(f"[runtime] Image built: {image.tags}")
        return self._image

    def _verify_agent_runtime(self, container: Any, agent_spec: AgentSpec) -> None:
        if agent_spec.adapter != "hermes":
            return

        result = container.exec_run(
            ["python", "-c", self._HERMES_PAST_BENCH_TOOLS_PROBE]
        )
        exit_code = getattr(result, "exit_code", None)
        output = getattr(result, "output", b"")
        if exit_code == 0:
            return

        detail = ""
        if output:
            if isinstance(output, bytes):
                detail = output.decode("utf-8", errors="replace").strip()
            else:
                detail = str(output).strip()
        if detail:
            detail = f" Details: {detail}"
        raise RuntimeError(
            f"Runtime image {self._image} is too old for Hermes benchmark tool injection; "
            "rebuild it with `past-bench build-image --kind runtime`." + detail
        )

    def _get_mapped_port(self, container) -> int:
        container.reload()
        port_key = f"{self._config.server_port}/tcp"
        bindings = container.ports.get(port_key)
        if not bindings:
            raise RuntimeError(f"No port binding found for {port_key}. Container ports: {container.ports}")
        return int(bindings[0]["HostPort"])

    def _wait_healthy(self, url: str, timeout: int = 15) -> None:
        import httpx

        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(0.3)
        raise RuntimeError(
            f"Runtime service not ready at {url} after {timeout}s"
            + (f": {last_exc}" if last_exc else "")
        )
