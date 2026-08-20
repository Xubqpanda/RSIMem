from __future__ import annotations

from dataclasses import dataclass

import pytest

from rsimem.memory import (
    MemoryAccessMode,
    MemoryArtifact,
    MemoryBackendDescriptor,
    MemoryBackendRegistry,
    MemoryExperience,
    MemoryKind,
    MemoryKindCapability,
    MemoryMessage,
    MemoryMutation,
    MemoryMutationAction,
    MemoryMutationResult,
    MemoryResource,
)


@dataclass
class _Backend:
    name: str
    kind: MemoryKind

    @property
    def descriptor(self) -> MemoryBackendDescriptor:
        return MemoryBackendDescriptor(
            self.name,
            (MemoryKindCapability(self.kind, MemoryAccessMode.SEARCH),),
        )

    def get(self, artifact_id: str):
        return None

    def query(self, query):
        return ()

    def mutate(self, mutation: MemoryMutation) -> MemoryMutationResult:
        return MemoryMutationResult(True, self.name, mutation.action)

    def close(self) -> None:
        return None


class _Compiler:
    name = "test-compiler"
    output_kinds = frozenset({MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL})

    def compile(self, experience: MemoryExperience) -> tuple[MemoryMutation, ...]:
        fact = MemoryArtifact(
            artifact_id=f"fact:{experience.experience_id}",
            kind=MemoryKind.SEMANTIC,
            content="The deployment requires Python 3.11.",
        )
        skill = MemoryArtifact(
            artifact_id=f"skill:{experience.experience_id}",
            kind=MemoryKind.PROCEDURAL,
            title="deploy-service",
            content="---\nname: deploy-service\ndescription: Deploy the service\n---\nRun tests first.",
            resources=(MemoryResource("scripts/check.sh", b"pytest -q\n"),),
        )
        return (
            MemoryMutation(MemoryMutationAction.ADD, fact.kind, artifact=fact),
            MemoryMutation(MemoryMutationAction.ADD, skill.kind, artifact=skill),
        )


def test_standard_taxonomy_and_string_values_are_normalized() -> None:
    assert {item.value for item in MemoryKind} == {
        "semantic",
        "episodic",
        "procedural",
    }
    artifact = MemoryArtifact("fact-1", "semantic", "A durable fact.")
    mutation = MemoryMutation("add", "semantic", artifact=artifact)
    assert artifact.kind == MemoryKind.SEMANTIC
    assert mutation.action == MemoryMutationAction.ADD


def test_mutation_invariants_reject_ambiguous_operations() -> None:
    semantic = MemoryArtifact("fact-1", MemoryKind.SEMANTIC, "A durable fact.")
    episodic = MemoryArtifact("episode-1", MemoryKind.EPISODIC, "A completed task.")

    with pytest.raises(ValueError, match="requires artifact"):
        MemoryMutation(MemoryMutationAction.ADD, MemoryKind.SEMANTIC)
    with pytest.raises(ValueError, match="kind must match"):
        MemoryMutation(MemoryMutationAction.ADD, MemoryKind.SEMANTIC, artifact=episodic)
    with pytest.raises(ValueError, match="does not accept"):
        MemoryMutation(
            MemoryMutationAction.ADD,
            MemoryKind.SEMANTIC,
            artifact=semantic,
            artifact_id="old-fact",
        )
    with pytest.raises(ValueError, match="requires artifact_id"):
        MemoryMutation(MemoryMutationAction.DELETE, MemoryKind.SEMANTIC)
    with pytest.raises(ValueError, match="instead of artifact"):
        MemoryMutation(
            MemoryMutationAction.DELETE,
            MemoryKind.SEMANTIC,
            artifact=semantic,
        )


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "scripts/../../secret", "SKILL.md"])
def test_procedural_resource_paths_cannot_escape_artifact(path: str) -> None:
    with pytest.raises(ValueError, match="resource path"):
        MemoryResource(path, b"secret")


def test_only_procedural_memory_can_bundle_resources() -> None:
    with pytest.raises(ValueError, match="procedural memory"):
        MemoryArtifact(
            "fact-1",
            MemoryKind.SEMANTIC,
            "A durable fact.",
            resources=(MemoryResource("references/source.md", b"source"),),
        )


def test_compiler_turns_an_episode_into_typed_mutations() -> None:
    experience = MemoryExperience(
        experience_id="episode-7",
        session_id="session-3",
        task_id="task-1",
        outcome="success",
        messages=(
            MemoryMessage("user", "Deploy the service."),
            MemoryMessage("assistant", "Deployment completed."),
        ),
    )
    compiler = _Compiler()
    mutations = compiler.compile(experience)

    assert compiler.output_kinds == {MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL}
    assert [mutation.kind for mutation in mutations] == [
        MemoryKind.SEMANTIC,
        MemoryKind.PROCEDURAL,
    ]
    assert mutations[1].artifact is not None
    assert mutations[1].artifact.resources[0].path == "scripts/check.sh"


def test_registry_registration_is_atomic_on_route_conflict() -> None:
    registry = MemoryBackendRegistry()
    first = _Backend("semantic-a", MemoryKind.SEMANTIC)
    rejected = _Backend("semantic-b", MemoryKind.SEMANTIC)
    registry.register(first)

    with pytest.raises(ValueError, match="already routed"):
        registry.register(rejected)

    assert registry.resolve(MemoryKind.SEMANTIC) is first
    with pytest.raises(KeyError, match="unknown memory backend"):
        registry.select(MemoryKind.SEMANTIC, rejected.name)
