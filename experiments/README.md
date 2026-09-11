# Experiment Protocols

This directory owns repository-level, paper-specific experiment protocols:
frozen manifests, provider runners, aggregation, and historical replay.

The dependency direction is one-way: `experiments/` may import `rsimem`, but
package code under `src/rsimem/` must not import this directory. Protocol
implementations will be migrated here incrementally; legacy `rsimem.*`
launchers remain compatibility shims until a breaking release.
