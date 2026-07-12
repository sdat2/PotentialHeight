# DRAFT — ADCIRC v55.02 + BO-driver image for a single GCP VM. UNTESTED.
# The riskiest step is building ADCIRC's netCDF-Fortran output against parallel HDF5;
# CHECK THE FORK'S OWN BUILD NOTES FIRST: github.com/sdat2/adcirc-v55.02
# (it carries "compilation settings" that may prescribe an exact recipe/preset).
#
# Design: one conda env `ws` provides BOTH the ADCIRC build toolchain (gfortran + MPICH +
# parallel netCDF-Fortran) AND the Python BO stack. MPICH (not OpenMPI) so mpirun has no
# root guard in the container. The Python netCDF4/h5netcdf come from pip wheels (bundled
# libs) and are independent of the conda netCDF used to link ADCIRC — two separate stacks,
# on purpose.
FROM mambaorg/micromamba:1.5.8

USER root

# --- 1. toolchain + ADCIRC build deps (conda-forge, MPICH-flavoured parallel netCDF-Fortran) ---
# apt-free on purpose: plain-HTTP apt breaks on some networks (Fastly 500s) and debian-slim
# has no CA certs for HTTPS apt; conda-forge (own CA bundle) provides everything, git included.
# cmake<4: ADCIRC v55 declares cmake_minimum_required(2.8.12) and CMake 4 dropped <3.5 compat.
# cxx-compiler: CMake enables CXX by default for project(adcirc) and validates mpicxx, whose
# underlying x86_64-conda-linux-gnu-c++ only exists with this package installed.
RUN micromamba create -y -n ws -c conda-forge \
      python=3.10 git \
      fortran-compiler c-compiler cxx-compiler "cmake<4" make \
      mpich "hdf5=*=mpi_mpich*" "netcdf-fortran=*=mpi_mpich*" \
    && micromamba clean -a -y
ENV MAMBA_ROOT_PREFIX=/opt/conda
# run subsequent RUNs inside the env (so mpif90 / $CONDA_PREFIX resolve)
SHELL ["micromamba", "run", "-n", "ws", "/bin/bash", "-o", "pipefail", "-c"]

# --- 2. build ADCIRC: adcprep (serial partitioner) + padcirc (MPI solver), netCDF output on ---
# Flags mirror the fork's own compile.sh (its known-good recipe), with the only changes vs
# ARCHER2 being: Cray ftn/cc -> conda mpif90/mpicc, and Cray parallel-netCDF -> conda netCDF.
# -fallow-argument-mismatch (gfortran>=10) is load-bearing. The 132-char line-length flags
# go via the DEDICATED Fortran_LINELENGTH_FLAG / ADDITIONAL_FLAGS_* vars, NOT
# CMAKE_Fortran_FLAGS: the fork's cmake/macros.cmake expands them unquoted, and an unset
# var vanishes -> set_target_properties "incorrect number of arguments" at configure.
# SWAN/GRIB2/ASWIP/UTILITIES targets dropped (no waves in this pipeline).
ARG ADCIRC_REPO=https://github.com/sdat2/adcirc-v55.02.git
# NB: the fork is a WRAPPER repo — the CMake project lives in its adcirc/ subdirectory
# (a real vendored tree, so a --depth 1 clone contains it). -S must point there.
RUN git clone --depth 1 "$ADCIRC_REPO" /opt/adcirc-src && \
    LL="-ffixed-line-length-132 -ffree-line-length-132" && \
    cmake -S /opt/adcirc-src/adcirc -B /opt/adcirc-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_ADCPREP=ON -DBUILD_PADCIRC=ON \
      -DENABLE_OUTPUT_NETCDF=ON -DENABLE_DATETIME=ON \
      -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_Fortran_COMPILER=mpif90 \
      -DCMAKE_Fortran_FLAGS="-DREAL8 -DCSCA -DLINUX -g -fbacktrace -w -fallow-argument-mismatch -O2" \
      -DFortran_LINELENGTH_FLAG="$LL" \
      -DADDITIONAL_FLAGS_ADCPREP="$LL" \
      -DNETCDFHOME="$CONDA_PREFIX" \
      -DNETCDF_LIBRARY="$CONDA_PREFIX/lib/libnetcdf.so" \
      -DNETCDF_INCLUDE_DIR="$CONDA_PREFIX/include" \
      -DNETCDF_F90_LIBRARY="$CONDA_PREFIX/lib/libnetcdff.so" && \
    cmake --build /opt/adcirc-build -j "$(nproc)" && \
    mkdir -p /opt/adcirc/work && \
    find /opt/adcirc-build \( -name adcprep -o -name padcirc \) -type f -exec cp {} /opt/adcirc/work/ \; && \
    test -x /opt/adcirc/work/adcprep && test -x /opt/adcirc/work/padcirc && \
    ls -l /opt/adcirc/work   # assertion FAILS the layer if find copied nothing

# --- 3. worstsurge repo + BO Python stack (pip layer; trieste + TF 2.15 + hydra + netCDF4 ...) ---
# (repo renamed to sdat2/PotentialHeight; the old URL 301-redirects and git follows it)
RUN git clone --depth 1 https://github.com/sdat2/worstsurge.git /opt/worstsurge && \
    cd /opt/worstsurge && pip install -e ".[bo]" && pip install sithom && \
    mkdir -p /work/exp && chmod -R 777 /work   # writable run dir even without a -v mount

# --- 4. shims + smoke test: neutralise the two ARCHER2-isms; make the image self-testing ---
COPY shims/srun   /usr/local/bin/srun
COPY shims/module /usr/local/bin/module
COPY smoke_test.sh /usr/local/bin/adcirc-smoke-test
RUN chmod +x /usr/local/bin/srun /usr/local/bin/module /usr/local/bin/adcirc-smoke-test

# --- 5. runtime env ---
# ADCIRC_EXE_PATH + ADCIRC_NP + WORSTSURGE_MODULES(="" disables `module load`) are read
# by adforce.wrap.get_default_config(), which is how the ARGPARSE entry points
# (adbo.exp_*, adbo.sweep_vmax) pick up machine-local settings — Hydra dot-overrides
# don't reach them. ADCIRC_NP=16 is a SAFE DEFAULT (this image is never ARCHER2, whose
# 108-rank default would silently oversubscribe a VM); match it to the VM's vCPUs with
# `docker run -e ADCIRC_NP=<n>`.
ENV WORSTSURGE_DATA_ROOT=/work/data \
    WORSTSURGE_EXP_DIR=/work/exp \
    ADCIRC_EXE_PATH=/opt/adcirc/work \
    ADCIRC_NP=16 \
    WORSTSURGE_MODULES="" \
    PATH=/usr/local/bin:/opt/conda/envs/ws/bin:$PATH
WORKDIR /opt/worstsurge
# the actual command (materiality run or a BO experiment) is passed by run_on_gcp.sh; see README.
CMD ["bash"]
