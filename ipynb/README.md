# JUPYTER Notebooks



```python
# Convenient jupyter setup
%load_ext autoreload
%autoreload 2
%config IPCompleter.greedy=True
import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import datatree as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sithom.plot import plot_defaults, label_subplots, lim
from tcpips.constants import DATA_PATH, FIGURE_PATH
plot_defaults()
```

```bash
jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8899 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8811 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8822 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8833 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8844 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8855 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8866 --no-browser

jupyter-lab --ip 0.0.0.0 --port 8877 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8878 --no-browser


http://127.0.0.1:8811/lab?token=3902366e4e2dad2391d48c5c7e70850cdbcc305b814e65f
http://127.0.0.1:8822/lab?token=2c5d170aa1ce1e7facb789b698248bf023687f0522df5dca


```