from datetime import timedelta, datetime
import os
from adcircpy import AdcircMesh, AdcircRun #, Tides
from .constants import SETUP_PATH
# from adcircpy.forcing.winds import BestTrackForcing
#from adcircpy.server import SlurmConfig

# load an ADCIRC mesh grid from a `fort.14` file to a new mesh object
mesh = AdcircMesh.open(os.path.join(SETUP_PATH, "fort.14.mid"), crs="epsg:4326")

# add nodal attributes from a `fort.13` file to the mesh object
mesh.import_nodal_attributes(os.path.join(SETUP_PATH, "fort.13.mid"))

# create a tidal forcing object, using all constituents
# tidal_forcing = Tides()
# tidal_forcing.use_all() # get rid of tides.

# add data from the tidal forcing object to the mesh object
# mesh.add_forcing(tidal_forcing)

# create a wind forcing object for Hurricane Sandy (2012)
# wind_forcing = BestTrackForcing("Sandy2012")


# add wind forcing data to the mesh object
#mesh.add_forcing(wind_forcing)

# create a Slurm (HPC job manager) configuration object.

# create an ADCIRC run driver object
driver = AdcircRun(
    mesh=mesh,
    start_date=datetime(year=2004, month=9, day=1),
    end_date=datetime(year=2004, month=9, day=15),
    # server_config=slurm,
    spinup_time=timedelta(days=15),
    #output_interval=timedelta(hours=1),
    #output_variables=["zeta", "u", "v"],
    # output_netcdf=True,
    # output_directory=SETUP_PATH,
    # output_prefix="test_adpy_",
)
driver.timestep = 1.0
# driver.
# mesh.generate_tau0()


# write configuration files to the specified directory
driver.write(output_directory=os.path.join(SETUP_PATH, "test_adpy_sandy2012"), overwrite=True)

# python -m adforce.adpy
