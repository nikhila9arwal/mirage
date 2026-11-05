# Source provenance

This publishing checkout preserves the original Mirage history. Its import
base is `700396ec0fa501fe4dbbe7c06dc115f1bc76c7df` (the local Patch E checkpoint),
which descends from upstream MPK commit
`ca58a02d110a278bffaed503be5e66494db2597d`. The Patch E checkpoint is not itself
an upstream commit.

The FIFO baseline source was copied from:

```text
/u/elu1/mpk_selector_fifo_20261104T220853Z/checkout
```

`source_baseline.json` records SHA-256 hashes for every imported core source
file. Missing files in the curated snapshot are not deletions: upstream build
files, license, submodule pins, and documentation are retained.

Experiment programs are under `research/nas`, with their source-root lookup
changed to this repository. The current measurement entry point requires
explicit external fixture paths, and its Slurm launcher derives the working
directory from where it is submitted. These are packaging changes only.
The standalone test runner uses CXX/NVCC from the environment. Original site
launchers are archived; no raw traces or model weights are included.

The tag `nas-fifo-baseline` preserves the imported baseline and this packaging.
To inspect it without disturbing current work:

```bash
git worktree add --detach ../Async-NAS-F1-fifo-baseline nas-fifo-baseline
```

All original experiment directories remain unchanged. The Git publishing copy
does not replace the source used by any queued or running Slurm job.

The following commit imports only the winner-loop optimization and its tests
and measurement changes from:

```text
/u/elu1/mpk_fifo_winner_20261105T004506Z/checkout
```

`source_latest.json` records all imported core hashes. They match that source
exactly; no runtime/compiler refactoring was done while preparing this repo.
Both new switches default to zero, so an existing build does not silently opt
in. The new measurement launcher explicitly selects them for comparisons.
