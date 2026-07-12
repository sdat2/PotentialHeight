#!/bin/bash
# DRAFT — "did the ADCIRC build actually work?" one-command check. The image bakes this
# script onto PATH as `adcirc-smoke-test`, so the canonical invocation is:
#   docker run --rm --shm-size=8g --cap-add=SYS_PTRACE -v ~/work:/work adcirc-ws adcirc-smoke-test [NP]
# (equivalently: bash /opt/worstsurge/rerun/adcirc/smoke_test.sh from the cloned repo)
# Graduated: toolchain -> binaries -> MPI launch -> a real end-to-end adforce.wrap run that
# must produce a finite max water level in maxele.63.nc. Exits non-zero on the first class of
# failure so you know which layer broke. A binary that compiles but segfaults on the mesh is
# caught at step 4, not step 2.
set -uo pipefail
EXE="${ADCIRC_EXE_PATH:-/opt/adcirc/work}"
NP="${1:-4}"
fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  PASS  $1"; else echo "  FAIL  $1"; fail=1; fi; }

echo "### 1. toolchain ###"
check "mpif90 present"             "command -v mpif90"
check "netcdf-fortran (nf-config)" "nf-config --version"
check "parallel netCDF"            "nc-config --has-parallel | grep -qi yes"
echo "  (info) HDF5/netCDF build strings — expect mpi_mpich:"; micromamba list -n ws 2>/dev/null | grep -E "^ *(hdf5|netcdf-fortran|mpich) " | sed 's/^/    /' || true

echo "### 2. binaries ###"
check "adcprep exists + executable" "test -x $EXE/adcprep"
check "padcirc exists + executable" "test -x $EXE/padcirc"
echo "  padcirc dynamic links (want mpich + netcdf, nothing stray):"; ldd "$EXE/padcirc" 2>/dev/null | grep -iE "mpi|netcdf|hdf5" | sed 's/^/    /'
check "padcirc links an MPI"        "ldd $EXE/padcirc | grep -qi mpi"
check "padcirc links netcdf"        "ldd $EXE/padcirc | grep -qi netcdf"

echo "### 3. MPI launch ($NP ranks) ###"
cat > /tmp/hello.f90 <<'EOF'
program p
  use mpi
  integer :: ierr, rank
  call mpi_init(ierr); call mpi_comm_rank(mpi_comm_world, rank, ierr)
  print *, 'hello from rank', rank
  call mpi_finalize(ierr)
end program
EOF
check "mpif90 compiles + mpirun -np $NP runs" "mpif90 /tmp/hello.f90 -o /tmp/hello && mpirun -np $NP /tmp/hello"

echo "### 4. end-to-end ADCIRC run (adforce.wrap smoke, keeps outputs) ###"
mkdir -p /work/exp
RUN=/work/exp/smoke
python -m adforce.wrap name=smoke use_slurm=false \
    files.exe_path="$EXE" files.exp_path=/work/exp files.low_storage=false \
    slurm.modules='' slurm.tasks_per_node="$NP" slurm.reserved_cpus=0 2>&1 | tail -25
check "maxele.63.nc produced" "test -f $RUN/maxele.63.nc"
# NB: plain xr.open_dataset cannot parse ADCIRC output (neta exists as both dim and
# variable) — must use the repo's loader, like adforce's own observe_max_point does.
ZETA=$(python -c "import numpy as np; from adforce.mesh import xr_loader; d=xr_loader('$RUN/maxele.63.nc'); v=float(np.nanmax(np.asarray(d['zeta_max'].values,dtype='float64'))); print(v if np.isfinite(v) and abs(v)<100 else 'BAD')" 2>/dev/null | tail -1 || echo "BAD")
if [ "$ZETA" != "BAD" ] && [ -n "$ZETA" ]; then echo "  PASS  finite max water level (max zeta = ${ZETA} m)"; else echo "  FAIL  finite max water level (got: ${ZETA:-nothing})"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "SMOKE TEST PASSED — ADCIRC build works end to end."; else echo "SMOKE TEST FAILED — fix the first FAIL above (lower layers first)."; fi
exit "$fail"
