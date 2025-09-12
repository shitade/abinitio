'''
Module to write cif file from Materials Project.

Usage:
    >> python3 <Path/to/write_cif.py> -p <PACKAGE> -i <ID>
'''

from pathlib import Path
from abinitio import HOME, common

def write_cif(id: int,
              package: str):
    '''
    Method to write cif file from Materials Project.

    Args:
        id (int): Material ID in Materials Project.
        package (str): Ab initio package.
    '''
    import os
    structure = common.get_structure_from_materials_project(id = id)
    mater = structure.reduced_formula
    materdir = HOME / package / mater
    # Make materdir if not exist.
    if not os.path.exists(path = materdir):
        print(f'## {materdir} does not exist. ##')
        os.makedirs(name = materdir)
        print(f'## Making {materdir} finished. ##')
    filename = mater + '.cif'
    filename = materdir / filename
    # Write cif file.
    structure.to(filename = filename)
    print(f'## Writing {filename} finished. ##')

def run_write_cif():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--package', type = Path, required = True, help = 'Ab initio package')
    parser.add_argument('-i', '--id', type = int, required = True, help = 'Material ID in Materials Project')
    args = parser.parse_args()
    write_cif(id = args.id, package = args.package)

if __name__ == '__main__':
    run_write_cif()
