"""Shared construction of the adforce wrap config from BO/sweep sample inputs.

Both the Bayesian-optimization objective (``adbo.exp``) and the 1D intensity
sweep (``adbo.sweep_vmax``) turn a sample dictionary (track parameters and,
in tradeoff mode, an intensity ``vmax``) into an adforce configuration. That
logic lives here once so the two paths cannot drift apart.

Kept free of tensorflow/trieste imports so the sweep can run without them.
"""

import os
from typing import Optional

from adforce.wrap import get_default_config


def build_wrap_config(
    cfg: dict,
    inputs: dict,
    tmp_dir: str,
    tradeoff_curve: Optional[object] = None,
):
    """Construct the adforce wrap config for one model evaluation.

    Args:
        cfg (dict): Experiment-level settings; requires ``obs_lon``,
            ``obs_lat``, ``resolution`` and (when ``tradeoff_curve`` is None)
            ``profile_name``.
        inputs (dict): Sample values keyed by input name. Recognized keys:
            ``angle``, ``trans_speed``, ``displacement``, ``vmax`` (tradeoff
            mode only); any other key is set directly on the tc config.
        tmp_dir (str): Run folder for this evaluation.
        tradeoff_curve (w22.tradeoff.TradeoffCurve, optional): When given,
            a CLE15 profile is generated at ``inputs["vmax"]`` on the
            size-intensity tradeoff curve and written to
            ``<tmp_dir>/profile.json`` (self-describing provenance; adforce
            accepts the direct .json path). Defaults to None.

    Returns:
        OmegaConf: The configured wrap config.
    """
    wrap_cfg = get_default_config()
    wrap_cfg.files.run_folder = tmp_dir
    # delete fort.22.nc after run
    wrap_cfg.files.low_storage = True
    if tradeoff_curve is not None:
        # storm on the size-intensity tradeoff curve: generate the CLE15
        # profile for this sampled intensity and store it in the run folder
        profile_json = os.path.join(tmp_dir, "profile.json")
        tradeoff_curve.profile(inputs["vmax"], out_path=profile_json)
        wrap_cfg.tc.profile_name.value = profile_json
    else:
        wrap_cfg.tc.profile_name.value = cfg["profile_name"]
    wrap_cfg.adcirc.attempted_observation_location.value = [
        cfg["obs_lon"],
        cfg["obs_lat"],
    ]
    wrap_cfg.adcirc["resolution"].value = cfg["resolution"]
    for inp in inputs:
        if inp == "vmax":
            # handled above via the tradeoff-curve profile; not a track
            # parameter in the adforce tc config
            continue
        if inp != "displacement":
            if inp == "trans_speed":
                wrap_cfg.tc["translation_speed"].value = inputs[inp]
            else:
                wrap_cfg.tc[inp].value = inputs[inp]
        if inp == "displacement":
            wrap_cfg.tc.impact_location.value = [
                cfg["obs_lon"] + inputs["displacement"],
                cfg["obs_lat"],
            ]
    return wrap_cfg
