import sys
from hydra import compose, initialize_config_dir
from adforce.fort22 import create_fort22
out, profile = sys.argv[1], sys.argv[2]
with initialize_config_dir(config_dir="/opt/worstsurge/adforce/config", version_base=None):
    cfg = compose(config_name="wrap_config", overrides=[
        "name=tide",
        f"tc.profile_name.value={profile}",
        "grid.Main.start=2005-08-19T00:00:00", "grid.Main.tlen=1249",
        "grid.TC1.start=2005-08-19T00:00:00", "grid.TC1.tlen=1249"])
    create_fort22(out, cfg.grid, cfg.tc)
print("fort22 written to", out)
