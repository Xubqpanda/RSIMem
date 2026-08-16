# MemBridge

MemBridge is the experiment and evaluation repository for the LightMem2 context-memory middleware. It measures the global cost of agentic memory systems while holding the underlying agent model fixed.

The initial experiment path is documented in [`docs/dataset_selection.md`](docs/dataset_selection.md). The first smoke test uses LoCoMo with MemBase, followed by LongMemEval and an interactive PAST-Bench integration.

Download the first local dataset with:

```bash
bash scripts/download_locomo.sh
```
