"""Generate one reservoir sample and package outputs for storage.

Wraps a GeoRules Layer construction and ``create_geology`` call with
deterministic seeding, binarises the resulting facies / active array,
and casts porosity and permeability to float16 for on-disk compactness.

``layer_type`` strings exposed by this module:

* ``lobe``       — :py:class:`georules.LobeLayer`
* ``gaussian``   — :py:class:`georules.GaussianLayer`
* ``meandering`` — :py:class:`georules.MeanderingChannelLayer` (Alluvsim-
  faithful fluvial engine; supports the canonical Pyrcz 2004 presets
  via the ``preset`` config field — see :func:`_apply_preset`)
* ``braided``    — :py:class:`georules.BraidedChannelLayer`
* ``delta``      — :py:class:`georules.DeltaLayer` (fluvial-engine
  delta with trunk-length / progradation / branch-spread / mouth-bar
  controls; see ``DELTA_FAN`` baseline)
"""

import numpy as np

import georules as gr
from .captions import caption_for


_LAYER_FACTORY = {
    "lobe": gr.LobeLayer,
    "gaussian": gr.GaussianLayer,
    "meandering": gr.MeanderingChannelLayer,
    "braided": gr.BraidedChannelLayer,
    "delta": gr.DeltaLayer,
}


def _apply_preset(params: dict) -> dict:
    """Resolve any ``"preset"`` entry in ``params`` to a baseline dict.

    If ``params["preset"]`` is set to one of the known preset names,
    the matching preset dict is loaded and ``params`` (excluding
    ``"preset"`` itself) is merged on top — i.e. user values override
    the preset baseline. Allows config-side syntax like::

        "params": {
            "preset": {"choices": ["PV_SHOESTRING", "CB_JIGSAW"]},
            "mCHsinu": {"range": [1.1, 1.6]}
        }

    so the dataset can mix five canonical Alluvsim architectures with
    only a handful of varying knobs on top.
    """
    preset_name = params.pop("preset", None)
    if preset_name is None:
        return params
    from georules.layers.channel import (
        PV_SHOESTRING, CB_JIGSAW, CB_LABYRINTH, SH_DISTAL, SH_PROXIMAL,
    )
    from georules.layers.delta import DELTA_FAN
    PRESETS = {
        "PV_SHOESTRING": PV_SHOESTRING,
        "CB_JIGSAW": CB_JIGSAW,
        "CB_LABYRINTH": CB_LABYRINTH,
        "SH_DISTAL": SH_DISTAL,
        "SH_PROXIMAL": SH_PROXIMAL,
        "DELTA_FAN": DELTA_FAN,
    }
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; "
            f"expected one of {sorted(PRESETS)}")
    out = dict(PRESETS[preset_name])
    out.update(params)
    return out


def generate_sample(job: dict, grid_cfg: dict):
    """Generate one sample.

    Returns ``(facies, poro, perm, meta)``:

    - ``facies`` — int8 array in {0, 1}, shape ``(nx, ny, nz)``
    - ``poro``   — float16 array, shape ``(nx, ny, nz)``
    - ``perm``   — float16 array (mD), shape ``(nx, ny, nz)``
    - ``meta``   — dict with ``layer_type``, ``seed``, ``caption`` and
      every sampled parameter.
    """
    seed = int(job["seed"])
    np.random.seed(seed)
    layer_type = job["layer_type"]
    params = _apply_preset(dict(job["params"]))
    layer = _LAYER_FACTORY[layer_type](**grid_cfg)
    layer.create_geology(**params)

    facies = _binarize(layer, layer_type).astype(np.int8)
    # Clip to float16-safe ranges before casting: poro to [0, 1], perm to
    # [0, 60000] mD. This also silently coerces any residual +inf from
    # extreme perm_std draws into 60000 (float16 max is 65504).
    poro = np.clip(
        np.asarray(layer.poro_mat, dtype=np.float32), 0.0, 1.0
    ).astype(np.float16)
    perm = np.clip(
        np.asarray(layer.perm_mat, dtype=np.float32), 0.0, 6e4
    ).astype(np.float16)

    meta = {
        "layer_type": layer_type,
        "seed": seed,
        "caption": caption_for(layer_type, params),
        **params,
    }
    return facies, poro, perm, meta


def _binarize(layer, layer_type: str) -> np.ndarray:
    # GaussianLayer has no ``facies`` attribute; use ``active`` (0/1).
    # All other layer types: fold any ``facies > 0`` into binary 1
    # (LobeLayer encodes the lobe generation index as int, not 0/1).
    if layer_type == "gaussian":
        return np.asarray(layer.active) > 0
    return np.asarray(layer.facies) > 0
