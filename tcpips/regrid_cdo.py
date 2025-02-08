import os
from sithom.time import timeit
from .constants import CONFIG_PATH, CDO_PATH, RAW_PATH
from .files import locker


def call_cdo(input_path: str, output_path: str) -> None:
    os.system(f"ncdump {input_path}")
    # delete the misnamed coordinates (xmip's fault?)
    os.system(
        f"ncks -x -v lon_verticies,lat_bounds,lon_bounds   {input_path} {output_path+'.tmp'}"
    )
    # remap bilinearly using cdo in silent mode
    os.system(
        f"cdo -s remapbil,{CONFIG_PATH}/halfdeg.txt {output_path+'.tmp'} {output_path} > /dev/null"
    )
    try:
        os.remove(f"{output_path + '.tmp'}")
    except Exception as e:
        print(e)
    os.system(f"ncdump {output_path}")


@timeit
@locker(CDO_PATH)
def regrid_cmip6_part(
    exp: str = "ssp585",
    typ: str = "ocean",
    model: str = "CESM2",
    member: str = "r4i1p1f1",
) -> None:
    print(f"exp:{exp} typ:{typ} model:{model} member:{member}")
    in_path = os.path.join(RAW_PATH, exp, typ, model, member) + ".nc"
    folder = os.path.join(CDO_PATH, exp, typ, model)
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, member) + ".nc"
    call_cdo(in_path, out_path)


if __name__ == "__main__":
    # python -m tcpips.regrid_new
    regrid_cmip6_part(exp="ssp585", typ="atmos", model="CESM2", member="r4i1p1f1")
