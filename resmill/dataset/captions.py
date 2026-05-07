"""Natural-language description of one generated reservoir sample.

Each caption is a short sentence summarising the key physics parameters of
the sample. Used as the ``caption`` column in the per-sample parquet
metadata table and as a conditioning signal for future
text-conditioned generative models.
"""


def caption_for(layer_type: str, params: dict) -> str:
    """Return one sentence describing the reservoir parameters.

    ``layer_type`` may be the bare family name (``"lobe"``,
    ``"channel"``, ``"delta"``) or the encoded ``"channel:PRESET"`` form
    saved in the slim parquet — we strip the preset suffix to look up
    the family-level template.
    """
    family = str(layer_type).split(":", 1)[0]
    fn = _DISPATCH.get(family)
    if fn is None:
        raise ValueError(f"no caption template registered for {layer_type!r}")
    return fn(params)


def _emit(parts, p, key, fmt):
    """Append ``fmt.format(p[key])`` to ``parts`` only if ``key`` is in ``p``."""
    if key in p:
        parts.append(fmt.format(p[key]))


def _emit_pair(parts, p, key_ave, key_std, prefix, suffix=""):
    """Append e.g. ``"radius 2300 ± 460 m"`` only if at least the mean is set."""
    if key_ave not in p:
        return
    s = f"{prefix} {p[key_ave]:.0f}"
    if key_std in p:
        s += f" ± {p[key_std]:.0f}"
    parts.append(s + suffix)


def _lobe(p):
    head = "Turbidite-lobe deposit"
    parts = []
    _emit(parts, p, "poro_ave", "mean porosity {:.2f}")
    if "perm_ave" in p:
        perm_mD = 10.0 ** float(p["perm_ave"])
        if "perm_std" in p:
            parts.append(
                f"mean permeability {perm_mD:.0f} mD "
                f"(log10-std {p['perm_std']:.2f})"
            )
        else:
            parts.append(f"mean permeability {perm_mD:.0f} mD")
    _emit(parts, p, "ntg", "realized net-to-gross {:.2f}")
    if "r_ave" in p:
        s = f"semi-minor radius {p['r_ave']:.0f}"
        if "r_std" in p:
            s += f" ± {p['r_std']:.0f}"
        s += " m"
        if "r_ave_cells" in p:
            s += f" ({p['r_ave_cells']:.1f} cells)"
        parts.append(s)
    if "r_major_m" in p:
        s = f"semi-major radius {p['r_major_m']:.0f} m"
        if "r_major_cells" in p:
            s += f" ({p['r_major_cells']:.1f} cells)"
        parts.append(s)
    _emit(parts, p, "asp", "aspect ratio {:.1f}")
    if "dh_ave" in p:
        s = f"thickness {p['dh_ave']:.1f}"
        if "dh_std" in p:
            s += f" ± {p['dh_std']:.1f}"
        s += " m"
        if "dh_ave_cells" in p:
            s += f" ({p['dh_ave_cells']:.1f} cells)"
        parts.append(s)
    _emit(parts, p, "bouma_factor", "Bouma factor {:.1f}")
    _emit(parts, p, "azimuth", "lobe-elongation azimuth {:.0f}°")
    if "upthinning" in p:
        parts.append(
            "thinning upwards" if p["upthinning"] else "without thinning upwards"
        )
    return f"{head} with " + ", ".join(parts) + "."


def _gaussian(p):
    parts = ["Heterogeneous sand-shale body from sequential Gaussian simulation"]
    _emit(parts, p, "poro_ave", "mean porosity {:.2f}")
    if "perm_ave" in p:
        perm_mD = 10.0 ** float(p["perm_ave"])
        if "perm_std" in p:
            parts.append(
                f"mean permeability {perm_mD:.0f} mD "
                f"(log10-std {p['perm_std']:.2f})"
            )
        else:
            parts.append(f"mean permeability {perm_mD:.0f} mD")
    _emit(parts, p, "ntg", "net-to-gross {:.2f}")
    _emit(parts, p, "nugget", "nugget {:.3f}")
    return ", ".join(parts) + "."


# Full prose names for each Pyrcz / Alluvsim channel preset. Spelled
# out per "User Guide to the Alluvsim Program" §5 (PV = paleo-valley,
# CB = channel and bar bodies, SH = sheet-sandstone) so captions read
# like a sentence a sedimentologist would write.
_CHANNEL_PRESET_NAMES = {
    "PV_SHOESTRING":   "Paleo-valley shoestring reservoir",
    "CB_JIGSAW":       "Channel-and-bar-bodies jigsaw reservoir",
    "CB_LABYRINTH":    "Channel-and-bar-bodies labyrinthine reservoir",
    "SH_DISTAL":       "Distal sheet-sandstone reservoir",
    "SH_PROXIMAL":     "Proximal sheet-sandstone reservoir",
    "MEANDER_OXBOW":   "Multi-storey meander-belt channel sandstone",
}


def _channel(p):
    preset = p.get("preset")
    head = _CHANNEL_PRESET_NAMES.get(preset, "Fluvial channel system")
    parts = []
    _emit(parts, p, "mCHsinu", "sinuosity {:.2f}")
    if "mCHdepth" in p:
        s = f"channel depth {p['mCHdepth']:.1f} m"
        if "mCHdepth_cells" in p:
            s += f" ({p['mCHdepth_cells']:.1f} cells)"
        parts.append(s)
    if "mCHwidth_m" in p:
        s = f"channel width {p['mCHwidth_m']:.0f} m"
        if "mCHwidth_cells" in p:
            s += f" ({p['mCHwidth_cells']:.1f} cells)"
        parts.append(s)
    _emit(parts, p, "mCHwdratio", "width:depth ratio {:.0f}")
    _emit(parts, p, "nlevel", "{:.0f} stacked levels")
    _emit(parts, p, "probAvulInside", "in-belt avulsion probability {:.2f}")
    _emit(parts, p, "mFFCHprop", "abandoned-mud fraction {:.2f}")
    _emit(parts, p, "ntg", "realized net-to-gross {:.2f}")
    _emit(parts, p, "azimuth", "flow azimuth {:.0f}°")
    return f"{head} with " + ", ".join(parts) + "."


def _delta(p):
    head = "Prograding distributary-fan delta"
    parts = []
    if "n_generations" in p:
        parts.append(f"{int(p['n_generations'])} stacked depositional generations")
    _emit(parts, p, "trunk_length_fraction", "trunk-length fraction {:.2f}")
    _emit(parts, p, "progradation_fraction", "progradation fraction {:.2f}")
    _emit(parts, p, "mCHsinu", "sinuosity {:.2f}")
    if "mCHdepth" in p:
        s = f"channel depth {p['mCHdepth']:.1f} m"
        if "mCHdepth_cells" in p:
            s += f" ({p['mCHdepth_cells']:.1f} cells)"
        parts.append(s)
    if "mCHwidth_m" in p:
        s = f"channel width {p['mCHwidth_m']:.0f} m"
        if "mCHwidth_cells" in p:
            s += f" ({p['mCHwidth_cells']:.1f} cells)"
        parts.append(s)
    _emit(parts, p, "mFFCHprop", "abandoned-mud fraction {:.2f}")
    _emit(parts, p, "ntg", "realized net-to-gross {:.2f}")
    if "paint_mouth_bars" in p:
        parts.append(
            "mouth bars on" if p["paint_mouth_bars"] else "mouth bars off"
        )
    _emit(parts, p, "azimuth", "flow azimuth {:.0f}°")
    return f"{head} with " + ", ".join(parts) + "."


_DISPATCH = {
    "lobe": _lobe,
    "gaussian": _gaussian,
    "channel": _channel,
    "delta": _delta,
}
