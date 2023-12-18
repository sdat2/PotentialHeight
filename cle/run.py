import os
import json
from sithom.io import read_json

print(read_json("inputs.json"))

# run octave file r0_pm.m
os.system("octave r0_pm.m")

# read in the output from r0_pm.m
print(read_json("outputs.json"))
