"""
Regrid CMIP6 data using CDO

This script provides functionality to regrid CMIP6 data using the Climate Data Operators (CDO) tool.
"""

import os
from sithom.time import timeit
from .constants import CONFIG_PATH, CDO_PATH, RAW_PATH
from .files import locker


def call_cdo(input_path: str, output_path: str) -> None:
    """
    Call cdo to regrid the input file to the output file.

    This function uses the cdo command line tool to regrid the input file
    to the output file. It first prints the header of the input file, then
    deletes the unnecessary coordinates, and finally remaps the file
    bilinearly using cdo. The output file is saved in the specified
    output path. The temporary file created during the process is deleted
    after the process is completed.

    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file.
    """
    os.system(f"ncdump -h {input_path}")
    # delete the misnamed coordinates (xmip's fault?)
    os.system(
        f"ncks -O -4 -x -v lon_verticies,lat_bounds,lon_bounds,lat_verticies,dcpp_init_year,member_id   {input_path} {output_path+'.tmp'}"
    )
    # relabel the standard names for lat and lon so that CDO can undertand them
    os.system(
        f"ncatted -O -a standard_name,lat,o,c,'latitude' -a standard_name,lon,o,c,'longitude' {output_path+'.tmp'}"
    )

    if "atmos" in input_path:
        typ = "atmos"
    elif "ocean" in input_path:
        typ = "ocean"
    else:
        typ = None

    if "HADGEM3-GC31-MM" in input_path:
        model = "HADGEM3-GC31-MM"
    else:
        model = None

    # remap bilinearly using cdo in silent mode
    if typ == "atmos" and model == "HADGEM3-GC31-MM":
        # os.system(f"ncatted -O -a coordinates,hus,d,, {output_path+'.tmp'}")
        os.system(f"ncatted -O -a coordinates,hus,m,s,'lat lon' {output_path+'.tmp'}")
        os.system(f"ncatted -O -a coordinates,hurs,m,s,'lat lon' {output_path+'.tmp'}")
        os.system(f"ncatted -O -a coordinates,psl,m,s,'lat lon' {output_path+'.tmp'}")
        os.system(f"ncatted -O -a coordinates,ta,m,s,'lat lon' {output_path+'.tmp'}")
        os.system(f"ncatted -O -a coordinates,tos,m,s,'lat lon' {output_path+'.tmp'}")
        # Fix time_bnds mismatch
        os.system(f"ncks -O -v time_bnds {output_path+'.tmp'}")
    elif typ == "ocean" and model == "HADGEM3-GC31-MM":
        #    # Remove broken coordinates attributes on tos
        # ncatted -O -a coordinates,tos,d,, file.nc
        # os.system(f"ncatted -O -a coordinates,tos,d,, {output_path+'.tmp'}")
        # # Point lat/lon bounds to actual vars or remove
        os.system(f"ncatted -O -a coordinates,tos,m,s,'lat lon' {output_path+'.tmp'}")
        # ncatted -O -a bounds,lat,d,, file.nc
        # os.system(f"ncatted -O -a bounds,lat,d,, {output_path+'.tmp'}")
        # # ncatted -O -a bounds,lon,d,, file.nc
        # os.system(f"ncatted -O -a bounds,lon,d,, {output_path+'.tmp'}")

        # # If you want bounds to exist, rename the *_verticies vars to *_vertices and update bounds attrs
        # ncrename -O -v lat_verticies,lat_vertices -v lon_verticies,lon_vertices file.nc
    print("About to regrid processed file with CDO")
    os.system(f"ncdump -h {output_path+'.tmp'}")

    os.system(
        f"cdo -f nc4 -s remapbil,{CONFIG_PATH}/halfdeg.txt -setgrid,{output_path+'.tmp'} {output_path+'.tmp'} {output_path}"
    )

    # f"cdo -f nc4 -s remapbil,{CONFIG_PATH}/era5_grid_from_file.txt {output_path+'.tmp'} {output_path} > /dev/null"
    try:
        os.remove(f"{output_path + '.tmp'}")
    except Exception as e:
        print(e)
    os.system(f"ncdump -h {output_path}")


@timeit
@locker(CDO_PATH)
def regrid_cmip6_part(
    exp: str = "ssp585",
    typ: str = "ocean",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
) -> None:
    """
    Regrid CMIP6 part of the experiment.
    This function regrids the CMIP6 part of the experiment using the
    CDO tool. It takes the experiment, type, model, and member as input
    parameters. It constructs the input and output paths based on these
    parameters and calls the CDO function to perform the regridding.

    Args:
        exp (str, optional): Experiment name. Defaults to "ssp585".
        typ (str, optional): Type of data. Defaults to "ocean".
        model (str, optional): Model name. Defaults to "CESM2".
        member (str, optional): Member name. Defaults to "r4i1p1f1".
    """
    print(f"exp:{exp} typ:{typ} model:{model} member:{member}")
    in_path = os.path.join(RAW_PATH, exp, typ, model, member) + ".nc"
    folder = os.path.join(CDO_PATH, exp, typ, model)
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, member) + ".nc"
    call_cdo(in_path, out_path)


if __name__ == "__main__":
    # python -m tcpips.regrid_cdo
    # regrid_cmip6_part(exp="ssp585", typ="ocean", model="CESM2", member="r4i1p1f1")
    # regrid_cmip6_part(exp="ssp585", typ="atmos", model="CESM2", member="r4i1p1f1")
    # regrid_cmip6_part(exp="historical", typ="ocean", model="CESM2", member="r4i1p1f1")
    # regrid_cmip6_part(exp="historical", typ="atmos", model="CESM2", member="r4i1p1f1")
    # regrid_cmip6_part(exp="historical", typ="ocean", model="CESM2", member="r10i1p1f1")
    # regrid_cmip6_part(exp="historical", typ="atmos", model="CESM2", member="r10i1p1f1")
    # regrid_cmip6_part(exp="historical", typ="ocean", model="CESM2", member="r11i1p1f1")
    # regrid_cmip6_part(exp="historical", typ="atmos", model="CESM2", member="r11i1p1f1")
    for model in ["CESM2"]:
        for member in ["r4i1p1f1", "r10i1p1f1", "r11i1p1f1"]:
            for exp in ["ssp585", "historical"]:
                for typ in ["ocean", "atmos"]:
                    regrid_cmip6_part(exp=exp, typ=typ, model=model, member=member)
