"""Slim-parquet column whitelist per layer family.

Two parquet files are written per shard:

* ``params.parquet`` — full schema, every meta key, for full reproducibility.
* ``params_slim.parquet`` — only the columns the flow-matching trainer
  consumes. Keep this list short and high-signal; everything dropped here
  remains recoverable from the full parquet.

Lookup is by ``layer_type`` *family* — channels with different presets
(``"channel:PV_SHOESTRING"`` etc.) share the channel column list because
the encoded ``layer_type`` value already discriminates architectures.
"""
from __future__ import annotations


# Universal columns every row carries.
_UNIVERSAL = [
    "layer_type",   # "lobe", "channel:PRESET", or "delta"
    "caption",
    "ntg",          # realized NTG over the cropped cube (active cell fraction)
    "poro_ave",     # realized mean porosity over active cells
    "perm_ave",     # realized mean log10(perm in mD) over active cells
    "azimuth",      # compass-CW degrees (0..360)
]

# Per-layer-family extras — concat with universal at lookup time.
_BY_FAMILY = {
    "lobe": [
        "width_cells",   # 2 × r_ave (full minor-axis diameter, cells)
        "depth_cells",   # dh_ave (lobe thickness, cells)
        "asp",           # major / minor aspect ratio
    ],
    "channel": [
        "width_cells",   # mCHwidth in cells
        "depth_cells",   # mCHdepth in cells
        "mCHsinu",
        "probAvulInside",
        "mFFCHprop",
    ],
    "delta": [
        "width_cells",
        "depth_cells",
        "mCHsinu",
        "probAvulInside",
        "mFFCHprop",
        "trunk_length_fraction",
    ],
}


def slim_columns(layer_type: str) -> list[str]:
    """Return the slim-parquet column whitelist for ``layer_type``.

    ``layer_type`` may be the bare family name (``"lobe"``, ``"channel"``,
    ``"delta"``) or the encoded ``"channel:PRESET"`` form — we strip the
    preset suffix to look up the family.
    """
    family = str(layer_type).split(":", 1)[0]
    if family not in _BY_FAMILY:
        raise KeyError(
            f"unknown layer_type family {family!r} (encoded={layer_type!r}); "
            f"expected one of {sorted(_BY_FAMILY)}")
    return _UNIVERSAL + _BY_FAMILY[family]
