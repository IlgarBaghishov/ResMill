"""Natural-language description of one generated reservoir sample.

Each caption is a short sentence summarising the key physics parameters of
the sample. Used as the ``caption`` column in the per-sample parquet
metadata table and as a conditioning signal for future
text-conditioned generative models.
"""


def caption_for(layer_type: str, params: dict) -> str:
    """Return one sentence describing the reservoir parameters."""
    fn = _DISPATCH.get(layer_type)
    if fn is None:
        raise ValueError(f"no caption template registered for {layer_type!r}")
    return fn(params)


def _lobe(p):
    # perm_ave is log10(mD) per LobeLayer.create_geology docstring.
    perm_mD = 10.0 ** float(p["perm_ave"])
    return (
        f"Turbidite lobe deposit with mean porosity {p['poro_ave']:.2f}, "
        f"mean permeability {perm_mD:.0f} mD (log10-std {p['perm_std']:.2f}), "
        f"net-to-gross {p['ntg']:.2f}, "
        f"lobe radius {p['rmin']:.0f}-{p['rmax']:.0f} m, "
        f"aspect ratio {p['asp']:.1f}, "
        f"thickness range {p['dhmin']}-{p['dhmax']} m, "
        f"bouma factor {p.get('bouma_factor', 0):.1f}, "
        f"upthinning {'enabled' if p.get('upthinning', True) else 'disabled'}."
    )


def _gaussian(p):
    # perm_ave is log10(mD) per GaussianLayer.create_geology docstring.
    perm_mD = 10.0 ** float(p["perm_ave"])
    return (
        f"Heterogeneous sand-shale body from sequential Gaussian simulation "
        f"with mean porosity {p['poro_ave']:.2f}, "
        f"mean permeability {perm_mD:.0f} mD (log10-std {p['perm_std']:.2f}), "
        f"net-to-gross {p['ntg']:.2f}, "
        f"nugget {p.get('nugget', 0.05):.3f}."
    )


def _meandering(p):
    return (
        f"Meandering fluvial channel system, "
        f"sinuosity {p.get('mCHsinu', 1.6):.2f}, "
        f"channel depth {p.get('mCHdepth', 2.5):.1f} m, "
        f"width:depth ratio {p.get('mCHwdratio', 10):.0f}, "
        f"avulsion-inside probability {p.get('probAvulInside', 0.05):.2f}, "
        f"abandoned-mud fraction {p.get('mFFCHprop', 0):.2f}, "
        f"NTG target {p.get('NTGtarget', 0.10):.2f}, "
        f"azimuth {p.get('azimuth', 0):.0f} deg."
    )


def _braided(p):
    return (
        f"Braided fluvial channel system, "
        f"sinuosity {p.get('mCHsinu', 1.3):.2f}, "
        f"channel depth {p.get('mCHdepth', 3.0):.1f} m, "
        f"width:depth ratio {p.get('mCHwdratio', 16):.0f}, "
        f"avulsion-inside probability {p.get('probAvulInside', 0.40):.2f}, "
        f"abandoned-mud fraction {p.get('mFFCHprop', 0.5):.2f}, "
        f"NTG target {p.get('NTGtarget', 0.30):.2f}, "
        f"azimuth {p.get('azimuth', 0):.0f} deg."
    )


def _delta(p):
    return (
        f"Distributary-fan delta with "
        f"{int(p.get('n_generations', 8))} stacked generations, "
        f"trunk length fraction {p.get('trunk_length_fraction', 0.4):.2f}, "
        f"progradation {p.get('progradation_fraction', 0):.2f}, "
        f"sinuosity {p.get('mCHsinu', 1.10):.2f}, "
        f"abandoned-mud fraction {p.get('mFFCHprop', 0):.2f}, "
        f"mouth bars {'on' if p.get('paint_mouth_bars', False) else 'off'}, "
        f"azimuth {p.get('azimuth', 0):.0f} deg."
    )


_DISPATCH = {
    "lobe": _lobe,
    "gaussian": _gaussian,
    "meandering": _meandering,
    "braided": _braided,
    "delta": _delta,
}
