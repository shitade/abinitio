'''
Module to deal with VASP input files.
Show available van der Waals functionals if executed.

Usage:
    >> python3 <Path/to/vasp.py>
'''

from pathlib import Path
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Incar, Kpoints, Poscar, Potcar
from abinitio import CLUSTER, common

# Pseudo potentials.
# potpaw.64 standard PBE potentials in https://www.vasp.at/wiki/index.php/Available_pseudopotentials .
_POTPAW = [
    'H', 'He',
    'Li_sv', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na_pv', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K_sv', 'Ca_sv', 'Sc_sv', 'Ti_sv', 'V_sv', 'Cr_pv', 'Mn_pv', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga_d', 'Ge_d', 'As', 'Se', 'Br', 'Kr',
    'Rb_sv', 'Sr_sv', 'Y_sv', 'Zr_sv', 'Nb_sv', 'Mo_sv', 'Tc_pv', 'Ru_pv', 'Rh_pv', 'Pd', 'Ag', 'Cd', 'In_d', 'Sn_d', 'Sb', 'Te', 'I', 'Xe',
    'Cs_sv', 'Ba_sv',
    'La', 'Ce', 'Pr_3', 'Nd_3', 'Pm_3', 'Sm_3', 'Eu_2', 'Gd_3', 'Tb_3', 'Dy_3', 'Ho_3', 'Er_3', 'Tm_3', 'Yb_2', 'Lu_3',
    'Hf_pv', 'Ta_pv', 'W_sv', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl_d', 'Pb_d', 'Bi_d', 'Po_d', 'At', 'Rn',
    'Fr_sv', 'Ra_sv', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm'
]

# Setting for van der Waals functionals.
_VDW = {
    'vdw-DF': {
        'GGA': 'RE', 'AGGAC': 0.0, 'LUSE_VDW': True, 'LASPH': True,
    },
    'vdW-DF2': {
        'GGA': 'ML', 'AGGAC': 0.0, 'LUSE_VDW': True, 'ZAB_VDW': -1.8867, 'LASPH': True
    },
    'optPBE-vdW': {
        'GGA': 'OR', 'AGGAC': 0.0, 'LUSE_VDW': True, 'LASPH': True
    },
    'optB88-vdW': {
        'GGA': 'BO', 'PARAM1': 0.1833333333, 'PARAM2': 0.22, 'AGGAC': 0.0, 'LUSE_VDW': True, 'LASPH': True
    },
    'optB86b-vdW': {
        'GGA': 'MK', 'PARAM1': 0.1234, 'PARAM2': 1.0, 'AGGAC': 0.0, 'LUSE_VDW': True, 'LASPH': True
    },
    'BEEF-vdW': {
        'GGA': 'BF', 'LUSE_VDW': True, 'ZAB_VDW': -1.8867, 'LASPH': True
    },
    'rev-vdW-DF2': {
        'GGA': 'MK', 'PARAM1': 0.1234568, 'PARAM2': 0.7114, 'AGGAC': 0.0, 'LUSE_VDW': True, 'ZAB_VDW': -1.8867, 'LASPH': True
    },
    'vdW-DF-cx': {
        'GGA': 'CX', 'AGGAC': 0.0, 'LUSE_VDW': True, 'LASPH': True
    },
    'vdW-DF3-opt1': {
        'GGA': 'BO', 'PARAM1': 0.1122334456, 'PARAM2': 0.1234568, 'AGGAC': 0.0, 'LUSE_VDW': True,
        'IVDW_NL': 3, 'ALPHA_VDW': 0.94950, 'GAMMA_VDW': 1.12, 'LASPH': True
    },
    'vdW-DF3-opt2': {
        'GGA': 'MK', 'PARAM1': 0.1234568, 'PARAM2': 0.58, 'AGGAC': 0.0, 'LUSE_VDW': True,
        'IVDW_NL': 4, 'ZAB_VDW': -1.8867, 'ALPHA_VDW': 0.28248, 'GAMMA_VDW': 1.29, 'LASPH': True
    },
    'rVV10': {
        'GGA': 'ML', 'LUSE_VDW': True, 'IVDW_NL': 2, 'BPARAM': 6.3, 'CPARAM': 0.0093, 'LASPH': True
    },
    'SCAN+rVV10': {
        'METAGGA': 'SCAN', 'LUSE_VDW': True, 'BPARAM': 15.7, 'CPARAM': 0.0093, 'LASPH': True
    },
    'PBE+rVV10L': {
        'GGA': 'PE', 'LUSE_VDW': True, 'BPARAM': 10, 'CPARAM': 0.0093, 'LASPH': True
    },
    'r2SCAN+rVV10': {
        'METAGGA': 'R2SCAN', 'LUSE_VDW': True, 'BPARAM': 11.95, 'CPARAM': 0.0093, 'LASPH': True
    }
}

_PACKAGE = 'vasp'

class IncarPoscar:
    '''
    Class of INCAR and POSCAR.

    Args:
        calculation (str): 'scf', 'nscf', 'bands'.
        incar (Incar):
        poscar (Poscar):

    Attributes:
        vdw (str): Van der Waals functional.
        lsorbit (bool): With or without spin-orbit coupling.
        magmom (list[float]): Magnetic moments [mu_{B}] of sites.
        potcar (Potcar):
        structure (Structure):
    '''
    def __init__(self,
                 calculation: str,
                 incar: Incar,
                 poscar: Poscar):
        self.calculation = calculation
        self.incar = incar
        self.poscar = poscar
    @property
    def vdw(self)-> str:
        '''
        Van der Waals functional.
        '''
        for (vdw, params) in _VDW.items():
            if all(self.incar.get(key = key) == (value.capitalize() if isinstance(value, str) else value)
                   for (key, value) in params.items()):
                return vdw
        return None
    @property
    def lsorbit(self)-> bool:
        '''
        With or without spin-orbit coupling.
        '''
        return self.incar.get(key = 'LSORBIT', default =  False)
    @property
    def magmom(self)-> list[float]:
        '''
        Magnetic moments [mu_{B}] of sites.
        '''
        if self.lsorbit and self.incar.get(key = 'MAGMOM') is not None:
            return [mom[2] for mom in self.incar['MAGMOM']]
        else:
            return self.incar.get(key = 'MAGMOM', default = [0] * sum(self.poscar.natoms))
    @property
    def potcar(self)-> Potcar:
        '''
        With recommended pseudopotentials.
        '''
        # Compressed atomic_numbers.
        atomic_numbers = sorted(set(self.structure.atomic_numbers), key = self.structure.atomic_numbers.index)
        # 1-based atomic number z to 0-based index z - 1.
        symbols = [_POTPAW[z - 1]
                   for z in atomic_numbers]
        return Potcar(symbols = symbols)
    @property
    def structure(self)-> Structure:
        return self.poscar.structure.add_site_property(property_name = 'magmom', values = self.magmom)
    @classmethod
    def from_file(cls,
                  vaspdir: Path,
                  calculation: str):
        '''
        Classmethod from INCAR and POSCAR.

        Args:
            vaspdir (Path): Working directory for VASP calculations.
            calculation (str): 'scf', 'nscf', 'bands'.
        '''
        # Read INCAR.
        filename = vaspdir / 'INCAR'
        incar = Incar.from_file(filename = filename)
        print(f'## Reading {filename} finished. ##')
        # Read POSCAR.
        filename = vaspdir / 'POSCAR'
        poscar = Poscar.from_file(filename = filename)
        print(f'## Reading {filename} finished. ##')
        return cls(calculation = calculation, incar = incar, poscar = poscar)
    @classmethod
    def from_structure(cls,
                       structure: Structure,
                       calculation: str,
                       lsorbit: bool,
                       nbands: int = None,
                       vdw: str = None,
                       prec: str = 'Accurate',
                       lmaxmix: int = 4,
                       ismear: int = 0,
                       sigma: float = 1e-2,
                       ediff: float = 1e-5,
                       **wannier90_win):
        '''
        Classmethod from Structure object.

        Args:
            structure (Structure):
            calculation (str): 'scf', 'nscf', 'bands'.
            lsorbit (bool): With or without spin-orbit coupling.
            nbands (int, optional): Number of bands. Defaults to None.
            vdw (str, optional): Van der Waals functional. Defaults to None.
            prec (str, optional): 'Normal', 'Single', 'SingleN', 'Accurate', 'Low', 'Medium', 'High'. Defaults to 'Accurate'.
            lmaxmix (int, optional): Maximum l-quantum number. Defaults to 4.
            ismear (int, optional): Smearing function. Defaults to 0 (Gaussian).
            sigma (float, optional): Smearing [eV]. Defaults to 1e-2.
            ediff (float, optional): Convergence criterion [eV]. Defaults to 1e-5.
            wannier90_win (dict[str]): Wannier90 parameters for vasp2wannier such as projections and exclude_bands.
                Use wannier90.Projections and wannier90.SpecifyBands, respectively.
        '''
        params = {'PREC': prec, 'ISMEAR': ismear, 'LMAXMIX': lmaxmix, 'EDIFF': ediff, 'SIGMA': sigma, 'SYSTEM': structure.reduced_formula}
        projections = wannier90_win.get('projections')
        num_wann = projections.num_wann if projections is not None else 0
        magmom = structure.site_properties['magmom']
        if lsorbit:
            params['LSORBIT'] = lsorbit
            # Flatten by sum(two_dimensional_list, []).
            params['MAGMOM'] = sum([[0, 0, mom]
                                    for mom in magmom], [])
            num_wann = 2 * num_wann
        else:
            # If magnetically ordered.
            if sum(map(abs, magmom)) > 0:
                params['ISPIN'] = 2
                params['MAGMOM'] = magmom
        exclude_bands = wannier90_win.get('exclude_bands')
        num_exclude = exclude_bands.num_specified if exclude_bands is not None else 0
        if nbands is not None:
            if nbands < num_wann + num_exclude:
                raise ValueError(f'nbands = {nbands} should be larger than or equal to sum of num_wann = {num_wann} and num_exlude = {num_exclude}.')
            params['NBANDS'] = nbands
        if vdw in _VDW.keys():
            params.update(**_VDW[vdw])
        else:
            vdw = None
        if calculation == 'bands':
            params['ICHARG'] = 11
            params['LCHARG'] = False
            params['LWAVE'] = False
        elif calculation == 'nscf':
            params['ICHARG'] = 11
            params['ISYM'] = -1
            params['LCHARG'] = False
            params['LWANNIER90'] = True
            lines = []
            if projections is not None:
                params['NUM_WANN'] = num_wann
                lines.append(str(projections))
            if exclude_bands is not None:
                lines.append(str(exclude_bands))
            params['WANNIER90_WIN'] = '\n'.join(['"'] + lines + ['"'])
        incar = Incar(params = params)
        poscar = Poscar(structure = structure, comment = structure.reduced_formula)
        return cls(calculation = calculation, incar = incar, poscar = poscar)
    @staticmethod
    def get_kpoints(kmesh: list[int],
                    shifts: list[int] = [0] * 3)-> Kpoints:
        '''
        Method to get Kpoints object.

        Args:
            kmesh (list[int]): Number of k-points for Monkhorst-Pack grids. Length is 3.
            shifts (list[int], optional): Shifts. Length is 3. Defaults to [0] * 3.

        Returns:
            Kpoints: Of Gamma-centered k-mesh.
        '''
        # Duplicate kmesh if only one component is specified.
        if isinstance(kmesh, int):
            kmesh = [kmesh]
        if len(kmesh) == 1:
            kmesh = kmesh * 3
        return Kpoints(kpts = [kmesh], kpts_shift = shifts)
    def copy_vdw(self,
                 vaspdir: Path):
        '''
        Method to copy vdw_kernel.bindat if necessary.

        Args:
            vaspdir (Path): Working directory for VASP caclulations.
        '''
        import shutil
        if self.vdw in _VDW.keys():
            # Copy vdw_kernel.bindat.
            shutil.copy2(src = CLUSTER['directories']['vasp_vdw'] / 'vdw_kernel.bindat', dst = vaspdir)
            print(f'## Copying {vaspdir / "vdw_kernel.bindat"} finished.')
    def get_jobscript(self,
                      queue_name: str,
                      num_procs: int,
                      pe_name: str = None)-> common.JobScript:
        '''
        Method to get common.JobScript object for vasp_std or vasp_ncl.

        Args:
            queue_name (str): Queue name.
            num_procs (int): Number of processors.
            pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

        Returns:
            common.JobScript:
        '''
        job_name = 'vasp'
        exe_name = job_name + ('_ncl' if self.lsorbit else '_std')
        exe_name = CLUSTER['directories'][_PACKAGE] / exe_name
        return common.JobScript(job_name = job_name, stdout = 'stdout', queue_name = queue_name,
                                num_procs = num_procs, exe_name = exe_name, pe_name = pe_name)
    def get_kpoints_line_mode(self,
                              kmesh: int)-> Kpoints:
        '''
        Method to get Kpoints object for band structure calculations.

        Arg:
            kmesh (int): Number of k-points per line.

        Returns:
            Kpoints: Of line mode.
        '''
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        # If three components of kmesh are specified, use first.
        if isinstance(kmesh, list):
            kmesh = kmesh[0]
        ibz = HighSymmKpath(structure = self.structure, has_magmoms = True, magmom_axis = [0, 0, 1])
        return Kpoints.automatic_linemode(divisions = kmesh, ibz = ibz)
    def write_incar(self,
                    vaspdir: Path):
        '''
        Method to write INCAR.

        Args:
            vaspdir: Working directory for VASP calculations.
        '''
        filename = vaspdir / 'INCAR'
        self.incar.write_file(filename = filename)
        print(f'## Saving {filename} finished. ##')
    def write_kpoints(self,
                      vaspdir: Path,
                      kmesh: list[int],
                      shifts: list[int] = [0] * 3):
        '''
        Method to write KPOINTS.

        Args:
            vaspdir: Working directory for VASP calculations.
            kmesh (list[int]): For bands, number of k-points per line.
                Otherwise, number of k-points for Monkhorst-Pack grids. Length is 3.
            shifts (list[int], optional): Shifts. Length is 3. Defaults to [0] * 3.
        '''
        if self.calculation == 'bands':
            kpoints = self.get_kpoints_line_mode(kmesh = kmesh)
        else:
            kpoints = self.get_kpoints(kmesh = kmesh, shifts = shifts)
        filename = vaspdir / 'KPOINTS'
        kpoints.write_file(filename = filename)
        print(f'## Saving {filename} finished. ##')
    def write_poscar(self,
                     vaspdir: Path):
        '''
        Method to write POSCAR.

        Args:
            vaspdir: Working directory for VASP calculations.
        '''
        filename = vaspdir / 'POSCAR'
        self.poscar.write_file(filename = filename)
        print(f'## Saving {filename} finished. ##')
    def write_potcar(self,
                     vaspdir: Path):
        '''
        Method to write POTCAR.

        Args:
            vaspdir: Working directory for VASP calculations.
        '''
        filename = vaspdir / 'POTCAR'
        self.potcar.write_file(filename = filename)
        print(f'## Saving {filename} finished. ##')

def get_efermi(vaspdir: Path)-> float:
    '''
    Method to get Fermi energy from OUTCAR.

    Args:
        vaspdir (Path): Working directory for VASP calculations.

    Returns:
        float: Fermi energy [eV].
    '''
    from pymatgen.io.vasp.outputs import Outcar
    # Read Outcar.
    filename = vaspdir / 'OUTCAR'
    outcar = Outcar(filename = filename)
    print(f'## Reading {filename} finished. ##')
    return outcar.efermi

def get_bandplotter(banddir: Path,
                    efermi: float)-> common.BandPlotter:
    '''
    Method to get common.BandPlotter object from vasprun.xml.

    Args:
        banddir (Path): Working directory for VASP band structure calculations.
        efermi (float): Fermi energy [eV].

    Returns:
        common.BandPlotter:
    '''
    from pymatgen.io.vasp.outputs import Vasprun
    # Read vasprun.xml.
    filename = banddir / 'vasprun.xml'
    vasprun = Vasprun(filename = filename, parse_potcar_file = False)
    print(f'## Reading {filename} finished. ##')
    bs = vasprun.get_band_structure(line_mode = True, efermi = 0)
    bandplot = common.BandPlotter.from_structure(structure = bs.structure)
    # Subtract Fermi energy.
    for eigvals in bs.bands.values():
        bandplot.add_bandstructure(bandstructure = common.BandStructure(distances = bs.distance, eigvals = eigvals - efermi))
    return bandplot

if __name__ == '__main__':
    print('## Available van der Waals functionals (case-sensitive): ##')
    print(*_VDW.keys())
