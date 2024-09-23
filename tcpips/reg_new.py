"""tcpips.reg_new: Regridding utilities for TCPIPS project."""

from typing import Union, Tuple
import numpy as np
import xarray as xr
import xesmf as xe


# Placeholder for 'can_coords' function from 'src.xr_utils'
def can_coords(da: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    """
    Ensure coordinate consistency and fix any irregularities.

    Args:
        da (Union[xr.Dataset, xr.DataArray]): Dataset or DataArray to process.

    Returns:
        Union[xr.Dataset, xr.DataArray]: Processed Dataset or DataArray.
    """
    # Placeholder implementation; replace with actual logic as needed.
    return da


def _grid_1d(
    start_b: float, end_b: float, step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 1D grid centers and bounds.

    Args:
        start_b (float): Start boundary (inclusive).
        end_b (float): End boundary (inclusive).
        step (float): Step size (grid resolution).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of centers and bounds arrays.
    """
    bounds = np.arange(start_b, end_b + step, step)
    centers = (bounds[:-1] + bounds[1:]) / 2
    return centers, bounds


def grid_1d(
    lon0_b: float,
    lon1_b: float,
    d_lon: float,
    lat0_b: float,
    lat1_b: float,
    d_lat: float,
) -> xr.Dataset:
    """
    Create a 1D rectilinear grid with specified longitude and latitude bounds and resolutions.

    Args:
        lon0_b (float): Lower longitude bound.
        lon1_b (float): Upper longitude bound.
        d_lon (float): Longitude step size (grid resolution).
        lat0_b (float): Lower latitude bound.
        lat1_b (float): Upper latitude bound.
        d_lat (float): Latitude step size (grid resolution).

    Returns:
        xr.Dataset: Dataset containing coordinate variables for the grid.
    """
    # Generate longitude centers and bounds, ensuring longitudes are within [0, 360)
    lon_centers, lon_bounds = _grid_1d(lon0_b, lon1_b, d_lon)
    lon_centers = lon_centers % 360
    lon_bounds = lon_bounds % 360

    # Generate latitude centers and bounds
    lat_centers, lat_bounds = _grid_1d(lat0_b, lat1_b, d_lat)

    ds = xr.Dataset(
        coords={
            "lon": (
                ["x"],
                lon_centers,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                ["y"],
                lat_centers,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon_b": (["x_b"], lon_bounds),
            "lat_b": (["y_b"], lat_bounds),
        }
    )

    return ds


def _regridding_ds_1d(with_bounds: bool = False) -> xr.Dataset:
    """
    Create a global 1D rectilinear grid for regridding.

    Args:
        with_bounds (bool, optional): Include bounds if True (required for conservative methods).
            Defaults to False.

    Returns:
        xr.Dataset: Dataset containing coordinate variables for regridding.
    """
    if with_bounds:
        # Include bounds for conservative regridding
        return grid_1d(-0.5, 359.5, 1.0, -90.0, 90.0, 1.0)
    else:
        # Exclude bounds; adjust latitude bounds to include poles
        ds = grid_1d(-0.5, 359.5, 1.0, -90.5, 90.5, 1.0)
        return ds.drop_vars(["lon_b", "lat_b"])


def regrid_2d(
    ds_input: Union[xr.Dataset, xr.DataArray],
    method: str = "bilinear",
    periodic: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Regrid a dataset or dataarray to a 1x1 degree global grid using 2D regridding.

    Args:
        ds_input (Union[xr.Dataset, xr.DataArray]): Input dataset or dataarray to regrid.
        method (str, optional): Regridding method ('bilinear', 'nearest_s2d', etc.).
            Defaults to 'bilinear'.
        periodic (bool, optional): Treat longitude as periodic. Defaults to True.

    Returns:
        Union[xr.Dataset, xr.DataArray]: Regridded dataset or dataarray.
    """
    # Create a global 1x1 degree target grid
    target_grid = xe.util.grid_global(1.0, 1.0)

    # Initialize the regridder
    regridder = xe.Regridder(
        ds_input,
        target_grid,
        method=method,
        periodic=periodic,
        ignore_degenerate=True,
        extrap_method="nearest_s2d",
    )

    # Perform the regridding
    ds_output = regridder(ds_input, keep_attrs=True)
    return ds_output


def regrid_2d_to_standard(
    da: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Adjust regridded data to standard coordinate names and format.

    Args:
        da (Union[xr.Dataset, xr.DataArray]): Regridded dataset or dataarray.

    Returns:
        Union[xr.Dataset, xr.DataArray]: Dataset or dataarray with standardized coordinates.
    """
    # Rename dimensions to standard names
    da = da.rename({"x": "X", "y": "Y"})

    # Assign coordinates based on regridded longitude and latitude values
    da = da.assign_coords(
        X=("X", da.isel(Y=0).lon.values % 360),
        Y=("Y", da.isel(X=0).lat.values),
    )

    # Drop unnecessary variables
    da = da.drop_vars(["lon", "lat"])

    # Ensure coordinates are consistent
    da = can_coords(da)

    # Sort by longitude for consistency
    da = da.sortby("X")
    return da


def regrid_1d(
    ds_input: Union[xr.Dataset, xr.DataArray],
    method: str = "bilinear",
    periodic: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Regrid a dataset or dataarray to a 1x1 degree global grid using 1D regridding.

    Args:
        ds_input (Union[xr.Dataset, xr.DataArray]): Input dataset or dataarray to regrid.
        method (str, optional): Regridding method ('bilinear', 'nearest_s2d', etc.).
            Defaults to 'bilinear'.
        periodic (bool, optional): Treat longitude as periodic. Defaults to True.

    Returns:
        Union[xr.Dataset, xr.DataArray]: Regridded dataset or dataarray.
    """
    # Create a 1D target grid without bounds
    target_grid = _regridding_ds_1d(with_bounds=False)

    # Initialize the regridder
    regridder = xe.Regridder(
        ds_input,
        target_grid,
        method=method,
        periodic=periodic,
        ignore_degenerate=True,
        extrap_method="nearest_s2d",
    )

    # Perform the regridding
    ds_output = regridder(ds_input, keep_attrs=True)
    return ds_output


def regrid_1d_to_standard(
    da: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Adjust regridded data to standard coordinate names and format.

    Args:
        da (Union[xr.Dataset, xr.DataArray]): Regridded dataset or dataarray.

    Returns:
        Union[xr.Dataset, xr.DataArray]: Dataset or dataarray with standardized coordinates.
    """
    # Rename dimensions to standard names
    da = da.rename({"x": "X", "y": "Y"})

    # Assign coordinates based on regridded longitude and latitude values
    da = da.assign_coords(
        X=("X", da.isel(Y=0).lon.values),
        Y=("Y", da.isel(X=0).lat.values),
    )

    # Drop unnecessary variables
    da = da.drop_vars(["lon", "lat"])

    # Ensure coordinates are consistent
    da = can_coords(da)
    return da


# Unit Tests
def test_grid_1d():
    lon0_b, lon1_b, d_lon = -0.5, 359.5, 1.0
    lat0_b, lat1_b, d_lat = -90.5, 90.5, 1.0
    ds = grid_1d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat)

    # Check that the coordinates exist
    assert "lon" in ds.coords
    assert "lat" in ds.coords

    # Check the lengths of the coordinate arrays
    assert len(ds.lon) == 360
    assert len(ds.lat) == 181

    # Check longitude range
    assert np.all(ds.lon.values >= 0) and np.all(ds.lon.values < 360)
    print("test_grid_1d passed")


def test_regrid_2d():
    # Create a sample dataset with 2x2 degree resolution
    ds_input = xe.util.grid_global(2.0, 2.0)
    ds_input["data"] = xr.DataArray(
        np.random.rand(len(ds_input["lat"]), len(ds_input["lon"])),
        dims=["lat", "lon"],
    )

    # Prepare dataset for regridding
    ds_input = ds_input.rename_dims({"lon": "x", "lat": "y"})
    ds_input = ds_input.transpose("y", "x")

    # Perform regridding
    ds_regridded = regrid_2d(ds_input)

    # Check dimensions
    assert ds_regridded.dims == ("Y", "X")
    print("test_regrid_2d passed")


def test_regrid_2d_to_standard():
    # Create a sample dataset and regrid it
    ds_input = xe.util.grid_global(2.0, 2.0)
    ds_input["data"] = xr.DataArray(
        np.random.rand(len(ds_input["lat"]), len(ds_input["lon"])),
        dims=["lat", "lon"],
    )
    ds_input = ds_input.rename_dims({"lon": "x", "lat": "y"})
    ds_input = ds_input.transpose("y", "x")
    ds_regridded = regrid_2d(ds_input)

    # Adjust to standard coordinates
    ds_standard = regrid_2d_to_standard(ds_regridded)

    # Check that standard coordinates exist
    assert "X" in ds_standard.coords
    assert "Y" in ds_standard.coords
    print("test_regrid_2d_to_standard passed")


def test_regrid_1d():
    # Create a sample dataset with 2x2 degree resolution
    ds_input = xe.util.grid_global(2.0, 2.0)
    ds_input["data"] = xr.DataArray(
        np.random.rand(len(ds_input["lat"]), len(ds_input["lon"])),
        dims=["lat", "lon"],
    )

    # Prepare dataset for regridding
    ds_input = ds_input.rename_dims({"lon": "x", "lat": "y"})
    ds_input = ds_input.transpose("y", "x")

    # Perform regridding
    ds_regridded = regrid_1d(ds_input)

    # Check dimensions
    assert ds_regridded.dims == ("Y", "X")
    print("test_regrid_1d passed")


def test_regrid_1d_to_standard():
    # Create a sample dataset and regrid it
    ds_input = xe.util.grid_global(2.0, 2.0)
    ds_input["data"] = xr.DataArray(
        np.random.rand(len(ds_input["lat"]), len(ds_input["lon"])),
        dims=["lat", "lon"],
    )
    ds_input = ds_input.rename_dims({"lon": "x", "lat": "y"})
    ds_input = ds_input.transpose("y", "x")
    ds_regridded = regrid_1d(ds_input)

    # Adjust to standard coordinates
    ds_standard = regrid_1d_to_standard(ds_regridded)

    # Check that standard coordinates exist
    assert "X" in ds_standard.coords
    assert "Y" in ds_standard.coords
    print("test_regrid_1d_to_standard passed")


if __name__ == "__main__":
    # Run unit tests
    test_grid_1d()
    test_regrid_2d()
    test_regrid_2d_to_standard()
    test_regrid_1d()
    test_regrid_1d_to_standard()
