#!/bin/bash
# DRAFT — VM bootstrap: install micromamba, build the env, clone the repo. UNTESTED.
# Runs either as the VM metadata `startup-script` (as root at boot) or by hand after SSH.
# Idempotent-ish: safe to re-run. Logs to /var/log/era5-startup.log when run as startup-script.
set -euo pipefail

# When run as GCE startup-script we're root with no HOME/login env; normalise to a real user.
# At first boot a stock Debian image may have NO uid-1000 login user yet (the guest agent
# creates accounts at first SSH), so fall back to creating a dedicated one.
RUN_USER="${SUDO_USER:-${USER:-$(whoami)}}"
if [ "$RUN_USER" = "root" ]; then
    RUN_USER="$(getent passwd 1000 | cut -d: -f1 || true)"
    if [ -z "$RUN_USER" ]; then
        id -u era5 >/dev/null 2>&1 || useradd -m -s /bin/bash era5
        RUN_USER=era5
    fi
fi
HOME_DIR="$(getent passwd "$RUN_USER" | cut -d: -f6)"
echo "bootstrapping for user=$RUN_USER home=$HOME_DIR"

run() { sudo -u "$RUN_USER" -H bash -lc "$*"; }

# 1. system deps
sudo apt-get update -y && sudo apt-get install -y git curl bzip2

# 2. micromamba (per-user). NB --strip-components=1 removes the leading bin/, so extract
# straight INTO ~/.local/bin (extracting to ~/.local would land it at ~/.local/micromamba).
run 'test -x ~/.local/bin/micromamba || (mkdir -p ~/.local/bin && \
     curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/.local/bin --strip-components=1 bin/micromamba)'
run 'grep -q MAMBA_ROOT ~/.bashrc || { echo "export MAMBA_ROOT_PREFIX=~/micromamba"; echo "export PATH=~/.local/bin:\$PATH"; echo "eval \"\$(micromamba shell hook -s bash)\""; } >> ~/.bashrc'

# 3. clone the repo (NB: repo renamed to sdat2/PotentialHeight; the old URL 301-redirects
# and git follows it, but update this if GitHub ever stops honouring the redirect)
run "test -d ~/worstsurge || git clone https://github.com/sdat2/worstsurge.git ~/worstsurge"

# 3b. point data/era5 at the local data dir BEFORE anything imports tcpips —
# tcpips/constants.py os.makedirs()es data/era5/{raw,pi_og,...} as REAL dirs at import
# time, which would otherwise block run_decade_gcs.sh's symlink later.
run 'mkdir -p ~/era5_data/era5 && { [ -L ~/worstsurge/data/era5 ] || rm -rf ~/worstsurge/data/era5; } && ln -sfn ~/era5_data/era5 ~/worstsurge/data/era5'

# 4. build the compute env from the checked-in environment.yml
run 'export PATH=~/.local/bin:$PATH MAMBA_ROOT_PREFIX=~/micromamba; \
     micromamba env list | grep -q worstsurge-era5 || \
     micromamba create -y -n worstsurge-era5 -f ~/worstsurge/rerun/era5/environment.yml'

# 5. validate the import chain (fail loudly if a dep is missing from environment.yml)
run 'export PATH=~/.local/bin:$PATH MAMBA_ROOT_PREFIX=~/micromamba PYTHONPATH=~/worstsurge; \
     micromamba run -n worstsurge-era5 python -c "import tcpips.era5, w22.ps, cle15.cle15n; print(\"import chain OK\")"'

echo "bootstrap done. SSH in and run:  cd ~/worstsurge/rerun/era5 && ./run_decade_gcs.sh <start> <end>"
