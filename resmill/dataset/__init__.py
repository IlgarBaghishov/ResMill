"""ResMill dataset generation subpackage.

Public entry points:

- ``resmill.dataset.build_jobs``     — expand a layers config into a flat job list
- ``resmill.dataset.generate_sample`` — produce one (facies, poro, perm, meta) tuple
- ``resmill.dataset.ShardWriter``    — buffered per-rank shard writer
- ``resmill.dataset.caption_for``    — natural-language description of one sample

The CLI (``python -m resmill.dataset.cli path/to/config.json``) combines
these with a SLURM_PROCID rank-stripe loop for multi-node HPC CPU
runs. See ``examples/dataset_generation/`` for configs and a submit script.
"""

from .captions import caption_for
from .generate import generate_sample
from .io import ShardWriter
from .sampling import build_jobs

__all__ = ["build_jobs", "generate_sample", "ShardWriter", "caption_for"]
