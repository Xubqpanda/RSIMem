"""ServiceManager — auto-start/stop mock services declared in task YAML."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from past_bench.models.task import ServiceDef


class ServiceStartError(RuntimeError):
    """Raised when a service fails to become ready within its timeout."""


class ServiceManager:
    """Context manager that ensures declared services are running.

    * On enter: for each service, probe its health-check endpoint.
      If unreachable, spawn the process and poll until ready (or timeout).
    * On exit: terminate any processes *we* spawned (reverse order).
    * ``reset_all()``: POST to each service's reset endpoint between trials.
    """

    def __init__(self, services: list[ServiceDef], cwd: Path | None = None) -> None:
        self._services = services
        self._cwd = cwd or Path.cwd()
        # Only processes we spawned ourselves — external ones are left alone.
        self._spawned: list[tuple[ServiceDef, subprocess.Popen]] = []  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ServiceManager:
        try:
            for svc in self._services:
                if self._is_healthy(svc):
                    print(f"  service '{svc.name}' already running on port {svc.port}")
                    continue
                self._spawn(svc)
        except Exception:
            # _spawn failed mid-way — kill already-spawned services to avoid port leaks
            self.__exit__(None, None, None)
            raise
        # Reset on entry so a new invocation doesn't inherit audit/state from a
        # prior process holding the same port.
        self.reset_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        for svc, proc in reversed(self._spawned):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"  service '{svc.name}' stopped")
        self._spawned.clear()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """POST to every service's reset endpoint (if declared)."""
        for svc in self._services:
            if svc.reset_endpoint:
                try:
                    httpx.post(svc.reset_endpoint, timeout=5.0)
                except Exception as exc:
                    print(f"  [WARN] reset failed for service '{svc.name}': {exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_healthy(self, svc: ServiceDef) -> bool:
        """Return True if the service responds to its health-check probe."""
        try:
            headers = {"X-Health-Check": "1"}
            method = svc.health_check_method.upper()
            # Bypass proxy for localhost health checks — mock services run locally.
            with httpx.Client(trust_env=False, timeout=2.0) as client:
                if method == "GET":
                    resp = client.get(svc.health_check, headers=headers)
                else:
                    resp = client.post(svc.health_check, headers=headers)
            return self._is_expected_health_response(svc, resp)
        except Exception:
            return False

    @staticmethod
    def _is_expected_health_response(svc: ServiceDef, resp: httpx.Response) -> bool:
        """Validate that the health probe matches the expected mock-service shape.

        Some developer machines already have unrelated services bound to the same
        ports as our mock services. A plain ``status_code < 500`` check can
        therefore silently accept the wrong server. We treat known mock-service
        endpoints semantically so a foreign 200 response does not mask the real
        service startup path.
        """
        if resp.status_code >= 500:
            return False

        try:
            payload = resp.json()
        except ValueError:
            payload = None

        # FastAPI validation errors are acceptable for readiness probes that hit
        # endpoints requiring a request body, e.g. ``/kb/search`` with no query.
        if resp.status_code == 422:
            return isinstance(payload, dict) and "detail" in payload

        path = urlparse(svc.health_check).path
        if path.endswith(("/health", "/documents/health")):
            return isinstance(payload, dict) and payload.get("status") == "ok"
        if path.endswith("/config/integrations"):
            return isinstance(payload, dict) and "integrations" in payload and "total" in payload
        if path.endswith("/notes/list"):
            return isinstance(payload, dict) and "notes" in payload and "total" in payload
        if path.endswith("/helpdesk/tickets"):
            return isinstance(payload, dict) and "tickets" in payload and "total" in payload
        if path.endswith("/kb/search"):
            return isinstance(payload, dict) and "articles" in payload and "total" in payload

        return resp.status_code < 500

    def _spawn(self, svc: ServiceDef) -> None:
        """Start the service subprocess and wait until healthy."""
        cmd = shlex.split(svc.command)
        # Replace bare "python" / "python3" with the current interpreter
        # so that forked workers (ProcessPoolExecutor) use the same venv.
        if cmd and cmd[0] in ("python", "python3"):
            cmd[0] = sys.executable
        # Build env: strip proxy vars to avoid routing mock traffic through proxies.
        base_env = dict(os.environ)
        for proxy_key in (
            "http_proxy", "https_proxy", "all_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        ):
            base_env.pop(proxy_key, None)
        env = {**base_env, **(svc.env or {})}
        proc = subprocess.Popen(
            cmd,
            cwd=self._cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        deadline = time.monotonic() + svc.ready_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                raise ServiceStartError(
                    f"Service '{svc.name}' exited immediately (rc={proc.returncode}): {stderr[:500]}"
                )
            if self._is_healthy(svc):
                self._spawned.append((svc, proc))
                print(f"  service '{svc.name}' started on port {svc.port}")
                return
            time.sleep(0.3)

        # Timed out — kill and report.
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        raise ServiceStartError(
            f"Service '{svc.name}' did not become ready within {svc.ready_timeout}s"
        )
