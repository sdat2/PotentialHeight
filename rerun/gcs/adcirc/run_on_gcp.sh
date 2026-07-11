#!/bin/bash
# DRAFT — provision a GCP VM, build the ADCIRC image, run the MATERIALITY CHECK, sync to GCS.
# UNTESTED. Read each step; run them by hand the first time. Reuses ../config.sh for
# PROJECT/ZONE/BUCKET; ADCIRC wants a bigger box than the ERA5 worker.
# PREREQUISITE: rerun/gcs/ AND the env-hook change (adforce/wrap.py +
# tests/test_wrap_config_env.py) must be COMMITTED AND PUSHED — both step 2's fresh clone
# on the VM and the image's internal `git clone` can only contain what's on origin.
# (`git status` must be clean for those paths; `git ls-tree origin/main rerun/gcs/` non-empty.)
set -euo pipefail
cd "$(dirname "$0")"
source ../config.sh

VM_ADCIRC="adcirc-worker"
MACHINE_ADCIRC="c2d-standard-16"     # 16 vCPU. Bump to -32/-56 to speed the full BO (below).
# MPI ranks: the image bakes a safe ADCIRC_NP=16 default; if you change MACHINE_ADCIRC,
# match it with `docker run -e ADCIRC_NP=<vCPUs>` (and the slurm.tasks_per_node override in
# the step-3 heredoc below). Small mesh (31k nodes) — don't over-partition; 16-32 is plenty.

echo "### 1. create the VM (on-demand for the first image build; spot is fine later) ###"
gcloud compute instances create "$VM_ADCIRC" \
    --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE_ADCIRC" \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=60GB --scopes=storage-rw

cat <<'REMOTE'
### 2. On the VM (gcloud compute ssh adcirc-worker --zone=...), one-time: ###
  sudo apt-get update && sudo apt-get install -y docker.io git
  sudo usermod -aG docker "$USER"   # re-login after this
  git clone https://github.com/sdat2/worstsurge.git ~/worstsurge
  cd ~/worstsurge/rerun/gcs/adcirc
  docker build -t adcirc-ws .       # builds ADCIRC v55.02 + BO env (~15-40 min first time)
  # (optional: push to Artifact Registry so you never rebuild:
  #   docker tag adcirc-ws $REGION-docker.pkg.dev/$PROJECT/ws/adcirc-ws && docker push ... )

### 3. MATERIALITY CHECK — the cheap, decisive first experiment (2 ADCIRC runs). ###
# Question: does the ~8% potential-size fix move the New Orleans surge enough to justify
# re-running the whole BO? Run the SAME idealized TC once with the OLD (pre-fix) CLE15
# profile and once with the FIXED profile, compare max surge at New Orleans.
#
# PROFILES — already generated + verified on the laptop (rerun/gcs/adcirc/profiles/:
# profile_{old,fixed}_{2015,2100}.json, PROVENANCE.json; produced by generate_profiles.py,
# wiring verified end-to-end by verify_wiring.py). Do NOT pass bare profile names inside
# the image: both eras share ONE canonical filename in w22/data, so names would silently
# run old-vs-old. fort22.py treats a profile_name ending in .json as a LITERAL PATH —
# these commands use that (same branch the adbo tradeoff sweep uses). Upload first:
#   gcloud compute scp rerun/gcs/adcirc/profiles/profile_*.json adcirc-worker:~/work/profiles/ --zone=$ZONE
  mkdir -p ~/work/exp ~/work/profiles
  for variant in old fixed; do
    docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v ~/work:/work adcirc-ws micromamba run -n ws \
      python -m adforce.wrap name=materiality_${variant}_2015 \
        tc.profile_name.value=/work/profiles/profile_${variant}_2015.json \
        files.exe_path=/opt/adcirc/work files.exp_path=/work/exp \
        slurm.modules='' slurm.tasks_per_node=16 slurm.reserved_cpus=0
  done
  # (repeat with _2100 for the end-of-century pair if the 2015 delta is material)
  # compare the max water level at New Orleans between the two run folders' maxele.63.nc
  # (adforce.wrap prints/returns it; or open ~/work/exp/materiality_*/maxele.63.nc).

REMOTE

# (outside the quoted heredoc so $BUCKET actually interpolates)
cat <<EOF
### 4. sync results off the disposable VM, then delete it ###
  gcloud storage cp -r ~/work/exp gs://$BUCKET/adcirc-exp/
EOF

cat <<EOF

### 5. FULL BO re-run (only if the materiality check says the fix matters) ###
  # The adbo.exp_*/sweep_vmax drivers are argparse, not Hydra — they pick up machine-local
  # settings from env vars read by adforce.wrap.get_default_config() (implemented + tested:
  # tests/test_wrap_config_env.py). The image bakes ADCIRC_EXE_PATH and WORSTSURGE_MODULES="";
  # you only pass the rank count for the VM: -e ADCIRC_NP=<vCPUs>. No yaml editing needed.
  #   docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v ~/work:/work -e ADCIRC_NP=16 adcirc-ws micromamba run -n ws \\
  #     python -m adbo.exp_3d --exp_name=new-orleans-2015-fixed \\
  #       --profile_name=2015_new_orleans_profile_r4i1p1f1 \\
  #       --obs_lat=29.9511 --obs_lon=-90.0715
  # Or the 1D size-intensity tradeoff sweep (curves in w22/data/curves/, plumbing-tested):
  #   docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v ~/work:/work -e ADCIRC_NP=16 adcirc-ws micromamba run -n ws \\
  #     python -m adbo.sweep_vmax --curve_nc w22/data/curves/2015_new_orleans_r4i1p1f1.nc \\
  #       --exp_name no-sweep-2015 --num 15
  # ~50 ADCIRC runs/BO experiment (15/sweep), sequential. Size the VM accordingly (see README).

  gcloud compute instances delete $VM_ADCIRC --zone=$ZONE   # tear down when done
EOF
