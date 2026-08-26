from types import SimpleNamespace

from past_bench.models.tool import ToolEndpoint
from past_bench.runner.agent_orchestrator import _runtime_metadata, _runtime_tool_endpoints


def test_runtime_metadata_preserves_join_identity_and_authoritative_trace() -> None:
    metadata = _runtime_metadata("trace-runtime", {
        "run_id": "run-sequence",
        "trace_id": "untrusted-trace",
        "episode_id": "episode-03",
        "family_id": "SM01",
        "stage": "eval_near",
    })

    assert metadata == {
        "run_id": "run-sequence",
        "trace_id": "trace-runtime",
        "episode_id": "episode-03",
        "family_id": "SM01",
        "stage": "eval_near",
    }


def test_runtime_tool_endpoints_adds_sandbox_endpoints_for_container_runtime():
    task = SimpleNamespace(
        tool_endpoints=[
            ToolEndpoint(
                tool_name="todo_create_task",
                url="http://localhost:9102/todo/tasks/create",
                method="POST",
            )
        ]
    )

    endpoints = _runtime_tool_endpoints(
        task,
        runtime_mode="container",
        sandbox_tools=True,
        sandbox_url="http://localhost:18080",
    )

    endpoint_map = {endpoint.tool_name: endpoint for endpoint in endpoints}
    assert endpoint_map["todo_create_task"].url == "http://host.docker.internal:9102/todo/tasks/create"
    assert endpoint_map["sandbox_shell_exec"].url == "http://host.docker.internal:18080/exec"
    assert endpoint_map["sandbox_file_read"].url == "http://host.docker.internal:18080/read"
    assert endpoint_map["sandbox_file_write"].url == "http://host.docker.internal:18080/write"
    assert endpoint_map["sandbox_browser_screenshot"].url == "http://host.docker.internal:18080/screenshot"


def test_runtime_tool_endpoints_leaves_local_runtime_urls_unchanged():
    task = SimpleNamespace(tool_endpoints=[])

    endpoints = _runtime_tool_endpoints(
        task,
        runtime_mode="local",
        sandbox_tools=True,
        sandbox_url="http://127.0.0.1:18080",
    )

    endpoint_map = {endpoint.tool_name: endpoint for endpoint in endpoints}
    assert endpoint_map["sandbox_shell_exec"].url == "http://127.0.0.1:18080/exec"
