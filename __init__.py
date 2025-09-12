'''
Module to deal with input/output files for VASP, Quantum Espresso, and Wannier90.

Before use, prepare setting.yaml.
In "queues", write names of queues and parallel environments. First parallel environment is default.
In "directories", write absolute paths.
Also, it is recommended to add MP_API_KEY for pymatgen to environment variable as
>> export MP_API_KEY="<MY_API_KEY>"

To use VASP, follow pymatgen installation:
https://pymatgen.org/installation.html
>> pmg config -p <EXTRACTED_VASP_POTCAR> <MY_PSP>
>> pmg config --add PMG_VASP_PSP_DIR <MY_PSP>
Directory may need to be renamed to POT_GGA_PAW_PBE and should be written in setting.yaml.

To use Quantum Espresso, download norm-conserving pseudopotentials from Pseudo Dojo:
https://www.pseudo-dojo.org/
Directories for scalar- and full-relativistic pseudopotentials should be written in setting.yaml.
Also, download dsjon files from Pseudo Dojo Github:
https://github.com/abinit/pseudo_dojo/tree/master/pseudo_dojo/pseudos/
Put them in respective directories.
'''

if __name__ == '__main__':
    help(__name__)

import os, yaml
from pathlib import Path

# Path to setting.yaml.
HOME = Path(os.environ['HOME'])
__CONF = HOME / 'abinitio' / 'setting.yaml'

with open(file = __CONF) as stream:
    CLUSTER = yaml.safe_load(stream = stream)
print(f'## Reading {__CONF} finished. ##')
for (key, value) in CLUSTER['directories'].items():
    CLUSTER['directories'][key] = Path(value)
