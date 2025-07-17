# Config folder for hydra

Main wrapping in `wrap_config.yaml`


Where to put the miscillanious run parameters? 

- experiment directory
- experiment name
- attempted observation location
- actual observation location

so there are three or four locations:

- attempted observation location
- actual observation location
- impact location
- (reference location to measure impact location from)

There is some redundancy as the impact location is on the line so that the TC passes through the impact location at impact time, so impact time and location could be changed simultaniously while staying on the same line.

## Problems with time

Time is specified in a few places:

- The start time `start` for the Main input grid/
    - The number of timesteps and size of timesteps determines the endtime.
- The start time `start` for the TC1 input grid.
    - The number of timesteps and size of timesteps determines the endtime.
- The NWS13ColdStartString parameter in the fort.15 file.
- The RNDAY parameter in fort.15.
- STATIM and REFTIM in fort.15.