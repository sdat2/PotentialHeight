# JUPYTER Notebooks



```python
# Convenient jupyter setup
%load_ext autoreload
%autoreload 2
%config IPCompleter.greedy=True
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sithom.plot import plot_defaults, label_subplots, lim
plot_defaults()
```

```bash
jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser
jupyter-lab --ip 0.0.0.0 --port 8899 --no-browser

```