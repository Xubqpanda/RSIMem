# Memory Adapter Architecture

## Purpose

RSIMem first separates memory representation and storage from the future LightRSI lifecycle policy. This allows experiments to replace a native memory framework without changing PAST-Bench tasks, graders, answer keys, or agent behavior.

The initial layer answers two implementation questions:

1. Which standard kind of memory is being read or changed?
2. Which backend owns that memory in the current experiment?

It does not yet decide whether evicted context deserves persistence. That decision belongs to the later LightRSI lifecycle controller.

## Standard Taxonomy

RSIMem uses the standard agent-memory meanings:

| Kind | Meaning | Typical examples |
|---|---|---|
| Semantic | Durable declarative knowledge independent of one event | Facts, user preferences, rules, constraints |
| Episodic | A situated experience tied to a session, task, or trajectory | Messages, actions, tool observations, outcomes |
| Procedural | Reusable knowledge about how to act | Skills, SOPs, workflows, scripts, templates |

These kinds describe meaning rather than storage layout. A vector database can hold all three kinds, while three files can implement three different kinds.

## Hermes Native Mapping

The first adapters expose Hermes' existing persistence without changing its model-visible behavior:

| RSIMem kind | Hermes substrate | Access mode | Mutation support |
|---|---|---|---|
| Semantic | `memories/MEMORY.md` and `memories/USER.md` | Eager system-prompt injection | Add, update, delete |
| Episodic | `state.db`, session messages, SQLite FTS5 | Search | Read-only through the RSIMem adapter |
| Procedural | `skills/**/SKILL.md` and supporting files | Progressive `skills_list` / `skill_view` disclosure | Add, update, delete |

The semantic adapter preserves Hermes' `memory` and `user` namespaces, `§` entry delimiter, native character budgets, and atomic file replacement. The procedural adapter preserves the skill directory layout and restricts resources to Hermes' `references`, `templates`, `scripts`, and `assets` roots. The episodic adapter intentionally remains read-only because Hermes owns session transcript creation and consistency.

The adapters do not yet invoke Hermes' prompt-injection scanner or skill security scanner. They are therefore an experiment substrate, not a permission boundary: untrusted compiler output must not be routed to mutation until host-neutral validation hooks are added in the integration stage.

## Backend And Compiler Roles

A `MemoryBackend` stores, retrieves, updates, and deletes already-typed memory artifacts. Hermes is the only backend in the current paper path. Mem0, LangMem, and similar projects may denote complete systems that combine construction policy with storage; in RSIMem, the Mem0 flat construction algorithm is separated from backend ownership and locally reimplemented as a semantic policy over Hermes storage.

A `MemoryCompiler` converts a completed `MemoryExperience` into zero or more typed `MemoryMutation` values. Text2Skill and SkillCreator belong primarily at this layer because their main responsibility is transforming experience into procedural knowledge. A compiler may target any backend selected for the resulting memory kind.

For the current PAST-Bench paper path, Hermes' native semantic, episodic, and procedural routes remain fixed, but only the semantic route receives a new policy implementation. The phase-two external contract is ingest/add rather than a model-selected mutation: the semantic policy may internally produce ADD, UPDATE, DELETE, or NONE, and RSIMem validates, applies, and accounts for that internal outcome. The first semantic policy locally reimplements Mem0's flat-memory construction algorithm from pinned MemBase source; MemBase itself is not imported at runtime. Episodic and procedural policy implementations remain deferred until their method-selection gates pass.

This distinction supports controlled comparisons:

- Hold the semantic policy fixed and replace the storage or retrieval backend in a future study.
- Hold the backend fixed and compare semantic construction/update methods in the current study.
- Hold routing and invocation fixed while comparing route-specific construction/update and retrieval policies under the same lifecycle-cost objective.

## Runtime Routing And Evidence

`MemoryBackendRegistry` selects exactly one backend per memory kind. `MemoryRuntime` routes queries and mutations and emits content-free lifecycle events for query, retrieval, injection, and mutation outcomes.

Events contain backend names, opaque artifact IDs, character counts, operation types, and reason codes. They do not contain query text, memory content, prompts, responses, credentials, or native provider payloads. This evidence can later be joined with the experiment ledger without making the ledger a second memory store.

## Current Boundary

Phase 2A fixes the three Hermes native routes in `rsimem.memory.ingestion`; only semantic has `policy_enabled=true`. Phase 2B extends that route-specific request with structured exit evidence, scope, validity, framework identity, canonical provenance, and content-free result/usage evidence. External callers still never submit ADD/UPDATE/DELETE/NONE or a target. `SemanticMemoryPolicy` returns host-neutral internal proposals, while the coordinator binds policy/framework/prompt/feature versions and resolves UPDATE/DELETE candidates to trusted artifact IDs and revisions. Mutating outcomes declare transaction and recovery-receipt requirements but are not executed. Disabled mode returns before policy resolution, preserving native routing and invocation.

The Phase 2B Mem0-flat package provides versioned prompt artifacts and a fixture-only completion client, and the ingestion layer provides a fixture-only deterministic pass-through policy. A separate append-only operation evidence contract can derive bounded atomic graphs offline without retaining prompt, response, query, source, or memory text. These are planning and observability contracts only: the semantic validation boundary and transactional executor remain disabled, and no Phase 2B path writes Hermes memory.

The adapter API is now available through an opt-in PAST-Bench Hermes bridge; direct native remains the default. Storage-boundary deterministic equivalence is complete for semantic rendering, episodic FTS views, and procedural resources. A second deterministic fixture invokes Hermes' real `AIAgent._build_system_prompt`, `session_search`, `skills_list`, and `skill_view` paths across `native`, `native+ledger`, and `native+adapter+ledger`. The session-search fixture replaces only the external summarizer with a deterministic function. It also verifies restart-stable artifact identity and explicit fail-closed or native-bypass behavior.

This establishes deterministic execution-surface equivalence, not matched
live-model PAST-Bench execution equivalence. A deterministic PAST-Bench
adapter-loop fixture now reaches all read surfaces and produces identical
final output across the three modes without making a model API call. Episode
runtime evidence is automatically joined into the ledger through comparison
identity validation.

The context lifecycle control plane now lives in [`lifecycle_controller.md`](lifecycle_controller.md). It decides when to evaluate a context snapshot and validates a joint context/memory signal, but it does not yet invoke compilers or route real mutations. The next implementation milestone is the Phase 2C host-neutral semantic validation and security boundary. Only after that gate and Phase 2D transaction/recovery pass should lifecycle signals route mutations through this runtime.
