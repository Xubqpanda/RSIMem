"""Load config.yaml with env-var expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


_ENV_RE = re.compile(r"\$\{(\w+)\}")

# Search order: CWD -> project root (where pyproject.toml lives)
_SEARCH_PATHS = [
    Path.cwd() / "config.yaml",
    Path(__file__).resolve().parent.parent.parent / "config.yaml",
]


def _expand_env(value: str) -> str | None:
    """Replace ${VAR} with os.environ[VAR]. Returns None if var is unset."""
    m = _ENV_RE.fullmatch(value.strip())
    if m:
        return os.environ.get(m.group(1))
    return value


def _walk_expand(obj):
    """Recursively expand ${ENV} references in string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    return obj


class ModelConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model_id: str = "anthropic/claude-opus-4-6"
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    system_prompt_prefix: str | None = None
    extra_body: dict | None = None


class JudgeConfig(BaseModel):
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = "google/gemini-2.5-flash"
    enabled: bool = True


class DefaultsConfig(BaseModel):
    trace_dir: str = "traces"
    tasks_dir: str = "past_bench_tasks"
    agent_registry: str = "configs/agents.yaml"


class SandboxConfig(BaseModel):
    """Configuration for Docker sandbox execution."""

    enabled: bool = False
    image: str = "past-bench-agent:latest"
    docker_host: str | None = None
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    sandbox_port: int = 8080
    container_timeout: int = 900
    max_concurrent: int = 10
    enable_browser: bool = True
    enable_shell: bool = True
    enable_file: bool = True


class RuntimeConfig(BaseModel):
    """Configuration for decoupled agent runtimes."""

    mode: str = "container"
    image: str = "past-bench-runtime:latest"
    temperature: float = 0.0
    docker_host: str | None = None
    server_port: int = 8090
    container_timeout: int = 900
    cache_dir: str = ".runtime_cache"
    registry_path: str = "configs/agents.yaml"
    host_alias: str = "host.docker.internal"


class PromptFilesConfig(BaseModel):
    """Workspace markdown files to inject into system prompt."""

    agents_md: str | None = None
    soul_md: str | None = None
    user_md: str | None = None
    tools_md: str | None = None


class SkillEntry(BaseModel):
    """A skill descriptor shown in the default skills list."""

    name: str
    description: str
    path: str


class SkillsConfig(BaseModel):
    """Skills configuration for prompt composition."""

    default: list[SkillEntry] = Field(default_factory=list)
    load_via_tool_call: bool = True
    read_tool_name: str = "read"


class BehaviorRulesConfig(BaseModel):
    """Behavior-policy text included in system prompt."""

    safety: str = "No independent objective; do not pursue self-preservation, replication, or resource acquisition."
    tool_call_style: str = "For low-risk actions, call tools directly without narration; narrate only for complex tasks."
    reply_tags: str = "Use [[reply_to_current]] to control reply relationship when needed."
    silent_reply: str = "If no reply is needed, output NO_REPLY."
    heartbeat: str = "Heartbeat checks should return HEARTBEAT_OK when no action is needed."


class PromptConfig(BaseModel):
    """Configuration for dynamic system prompt construction."""

    enabled: bool = True
    strict_file_check: bool = False
    include_tool_schema: bool = True
    files: PromptFilesConfig = PromptFilesConfig()
    behavior_rules: BehaviorRulesConfig = BehaviorRulesConfig()
    skills: SkillsConfig = SkillsConfig()


class MediaConfig(BaseModel):
    """Configuration for media detection and loading from prompts."""

    enabled: bool = True
    strict_mode: bool = False
    max_files: int = 6
    max_bytes_per_file: int = 8 * 1024 * 1024
    image_max_dimension: int = 2048


class Mem0Config(BaseModel):
    """mem0 semantic/episodic memory integration (skills/mem0/ package).

    When enabled, relevant memories are retrieved via vector search and
    injected into the system prompt before each task. After each task the
    message trace can be stored for future recall.

    Set via config.yaml:

        mem0:
          enabled: true
          api_key: ${MEM0_API_KEY}   # blank = self-hosted mode
          user_id: past_bench
    """

    enabled: bool = False
    api_key: str | None = None      # MEM0_API_KEY for cloud; blank = self-hosted
    host: str | None = None
    user_id: str = "past_bench"
    agent_id: str | None = None
    search_limit: int = 5           # memories to inject per task
    inject_memories: bool = True    # prepend memory block to system prompt
    learning_enabled: bool = True   # store trace after each task


class AcontextConfig(BaseModel):
    """Acontext skills/memory integration (skills/ package).

    When enabled, skills learned from previous runs are fetched from the
    configured learning space and injected into the system prompt.
    After each task the full message trace can be sent for distillation.

    Set via config.yaml:

        acontext:
          enabled: true
          api_key: ${ACONTEXT_API_KEY}
          space_id: <your-space-id>
    """

    enabled: bool = False
    api_key: str | None = None
    base_url: str | None = None
    space_id: str | None = None
    learning_enabled: bool = True   # trigger distillation after each task
    inject_skills: bool = True      # prepend skills block to system prompt


class LettaConfig(BaseModel):
    """Letta structured in-context memory integration (skills/Letta/ package).

    Requires a running Letta server (cloud or self-hosted Docker).
    See skills/Letta/README.md for setup.

    When enabled, the persistent PAST-Bench memory agent's MemoryBlocks
    are fetched and injected into the system prompt. After each task,
    the outcome can be sent back so the agent self-edits its blocks.

    Set via config.yaml:

        letta:
          enabled: true
          api_key: ${LETTA_API_KEY}
          agent_id: <memory-agent-id>
    """

    enabled: bool = False
    api_key: str | None = None     # LETTA_API_KEY (cloud) or server token
    base_url: str | None = None    # e.g. http://localhost:8283 for self-hosted
    agent_id: str | None = None    # persistent PAST-Bench memory agent ID
    inject_blocks: bool = True     # prepend MemoryBlocks to system prompt
    learning_enabled: bool = True  # send task outcomes back to the agent


class GraphitiConfig(BaseModel):
    """Graphiti temporal knowledge-graph integration (skills/Graphiti/ package).

    Requires a graph database backend (Neo4j, FalkorDB, or embedded Kuzu)
    and an OpenAI API key for entity extraction + embeddings.

    Set via config.yaml:

        graphiti:
          enabled: true
          backend: kuzu          # neo4j | falkordb | kuzu
          openai_api_key: ${OPENAI_API_KEY}
    """

    enabled: bool = False
    backend: str = "neo4j"              # neo4j | falkordb | kuzu
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    openai_api_key: str | None = None
    group_id: str = "past_bench"
    inject_graph: bool = True           # prepend graph context to system prompt
    learning_enabled: bool = True       # ingest task runs into the graph


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    judge: JudgeConfig = JudgeConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    sandbox: SandboxConfig = SandboxConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    prompt: PromptConfig = PromptConfig()
    media: MediaConfig = MediaConfig()
    acontext: AcontextConfig = AcontextConfig()
    mem0: Mem0Config = Mem0Config()
    letta: LettaConfig = LettaConfig()
    graphiti: GraphitiConfig = GraphitiConfig()


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file with ${ENV} expansion.

    Searches config.yaml in CWD then project root if path is not given.
    Returns defaults if no file is found.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = _SEARCH_PATHS

    for p in candidates:
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
            expanded = _walk_expand(raw)
            return Config.model_validate(expanded)

    return Config()
