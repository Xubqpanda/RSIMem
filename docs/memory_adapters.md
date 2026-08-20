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

A `MemoryBackend` stores, retrieves, updates, and deletes already-typed memory artifacts. Mem0, Graphiti, Letta, LangMem, and Hermes integrations belong behind this interface when they provide storage or retrieval.

A `MemoryCompiler` converts a completed `MemoryExperience` into zero or more typed `MemoryMutation` values. Text2Skill and SkillCreator belong primarily at this layer because their main responsibility is transforming experience into procedural knowledge. A compiler may target any backend selected for the resulting memory kind.

This distinction supports controlled comparisons:

- Hold the compiler fixed and replace the storage or retrieval backend.
- Hold the backend fixed and compare memory distillation methods.
- Let LightRSI choose whether compilation is worthwhile before paying its model and storage cost.

## Runtime Routing And Evidence

`MemoryBackendRegistry` selects exactly one backend per memory kind. `MemoryRuntime` routes queries and mutations and emits content-free lifecycle events for query, retrieval, injection, and mutation outcomes.

Events contain backend names, opaque artifact IDs, character counts, operation types, and reason codes. They do not contain query text, memory content, prompts, responses, credentials, or native provider payloads. This evidence can later be joined with the experiment ledger without making the ledger a second memory store.

## Current Boundary

The adapter API is available for isolated experiments, but the PAST-Bench runner still uses Hermes' native tools directly. This is intentional: the first milestone proves behavioral equivalence at the storage boundary before changing the runtime path.

The next integration milestone is an opt-in experiment configuration that selects adapters while preserving native Hermes as the baseline. Only after matched tests pass should LightRSI lifecycle signals invoke compilers and route their mutations through this runtime.
