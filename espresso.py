'''
Module to deal with Quantum Espresso input/output files.
'''

from io import TextIOWrapper
import numpy as np
from pathlib import Path
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from abinitio import CLUSTER, common, wannier90

# Directory of Pseudo Dojo pseudopotentials.
# Accuracy is standard or stringent.
_ACCURACY = 'standard'
# Hint is low, normal, or high.
_HINT = 'high'
_PSEUDO_DIR = {
    True: CLUSTER['directories']['espresso_fr'],
    False: CLUSTER['directories']['espresso_sr']
}
_JSON = {lspinorb: pseudo_dir / (_ACCURACY + '.djson')
         for (lspinorb, pseudo_dir) in _PSEUDO_DIR.items()}

_PACKAGE = 'espresso'

class Control:
    '''
    Class of &CONTROL namelist.

    Args:
        variables (dict[str]): Variables for &CONTROL namelist.
    '''
    __CHARACTERS = ['calculation', 'title', 'verbosity', 'restart_mode', 'outdir',
                    'wfcdir', 'prefix', 'disk_io', 'pseudo_dir']
    __LOGICALS = ['tstress', 'tprnfor', 'etot_conv_thr', 'forc_conv_thr', 'tefield',
                  'dipfield', 'lelfield', 'lorbm', 'lberry', 'gate',
                  'twochem', 'lfcp', 'trism']
    __INTEGERS = ['nstep', 'iprint', 'nberrycyc', 'gdir', 'nppstr']
    __REALS = ['dt', 'max_seconds']
    __NUMBERS = __INTEGERS + __REALS
    def __init__(self,
                 **variables):
        self.variables = variables
    def __str__(self):
        lines = [f'&{__class__.__name__.upper()}']
        for (key, value) in self.variables.items():
            if key in __class__.__CHARACTERS:
                lines.append(f'  {key} = \'{value}\',')
            elif key in __class__.__LOGICALS:
                lines.append(f'  {key} = .{value}.,')
            elif key in __class__.__NUMBERS:
                lines.append(f'  {key} = {value},')
        lines.append('/')
        return '\n'.join(lines)
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
        '''
        variables = {}
        while True:
            line = f.readline()
            # Stop when / is found.
            if line.startswith('/'):
                break
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            # Skip blank line.
            if not line:
                continue
            line = line.strip(',')
            # Variables are separated with =.
            line = line.split('=')
            key = line[0].strip()
            value = line[1].strip()
            if key in __class__.__CHARACTERS:
                variables[key] = value.strip("'")
            elif key in __class__.__LOGICALS:
                variables[key] = True if value.strip('.').lower().startswith('t') else False
            elif key in __class__.__INTEGERS:
                variables[key] = int(value)
            elif key in __class__.__REALS:
                variables[key] = float(value)
        return cls(**variables)

class System:
    '''
    Class of &SYSTEM namelist.

    Args:
        variables (dict[str]): Variables for &SYSTEM namelist.

    Attributes:
        lspinorb (bool): With or without spin-orbit coupling.
        nbnd (int): Number of bands.
        magmom (list[float]): Magnetic moments [mu_{B}] of sites.
    '''
    __CHARACTERS = ['occupations', 'smearing', 'pol_type', 'input_dft', 'exxdiv_treatment',
                    'dmft_prefix', 'constrained_magnetization', 'assume_isolated', 'esm_bc', 'vdw_corr']
    __LOGICALS = ['nosym', 'nosym_evc', 'noinv', 'no_t_rev', 'force_symmorphic',
                  'use_all_frac', 'one_atom_occupations', 'starting_spin_angle', 'sic_energy', 'noncolin',
                  'ace', 'x_gamma_extrapolation', 'dmft', 'ensemble_energies', 'lforcet',
                  'lspinorb', 'lgcscf', 'london', 'dftd3_threebody', 'ts_vdw_isolated',
                  'xdm', 'uniqueb', 'rhombohedral', 'relaxz', 'block']
    __INTEGERS = ['ibrav', 'nat', 'ntyp', 'nbnd', 'nbnd_cond',
                  'nr1', 'nr2', 'nr3', 'nr1s', 'nr2s',
                  'nr3s', 'nspin', 'nqx1', 'nqx2', 'nqx3',
                  'edir', 'report', 'esm_nfit', 'dftd3_version', 'space_group',
                  'origin_choice', 'nextffield']
    __REALS = ['celldm(1)', 'celldm(2)', 'celldm(3)', 'celldm(4)', 'celldm(5)',
               'celldm(6)', 'A', 'B', 'C', 'cosAB',
               'cosAC', 'cosBC', 'tot_charge', 'tot_magnetization', 'ecutwfc',
               'ecutrho', 'ecutfock', 'degauss_cond', 'nelec_cond', 'degauss',
               'sic_gamma', 'sci_vb', 'sci_cb', 'ecfixed', 'qcutz',
               'q2sigma', 'exx_fraction', 'screening_parameter', 'ecutvcut', 'localization_thr',
               'emaxpos', 'eopreg', 'eamp', 'lambda', 'esm_w',
               'esm_efield', 'gcscf_mu', 'gcscf_conv_thr', 'gcscf_beta', 'london_s6',
               'london_rcut', 'ts_vdw_econv_thr', 'xdm_a1', 'xdm_a2', 'zgate',
               'block_1', 'block_2', 'block_height']
    __REAL_LISTS = [['starting_charge','starting_magnetization', 'Hubbard_alpha', 'Hubbard_beta', 'angle1',
                     'angle2', 'fixed_magnetization', 'london_c6', 'london_rvdw'],
                    ['Hubbard_occ'],
                    ['starting_ns_eigenvalue']]
    __NUMBERS = __INTEGERS + __REALS
    def __init__(self,
                 **variables):
        self.variables = variables
    def __str__(self):
        lines = [f'&{__class__.__name__.upper()}']
        for (key, value) in self.variables.items():
            if key in __class__.__CHARACTERS:
                lines.append(f'  {key} = \'{value}\',')
            elif key in __class__.__LOGICALS:
                lines.append(f'  {key} = .{value}.,')
            elif key in __class__.__NUMBERS:
                lines.append(f'  {key} = {value},')
            # Flatten by sum(two_dimensional_list, []).
            elif key in sum(__class__.__REAL_LISTS, []):
                for (i, value_i) in enumerate(value):
                    if key in __class__.__REAL_LISTS[0]:
                        # 0-based index i to 1-based index i + 1.
                        lines.append(f'  {key}({i + 1}) = {value_i},')
                    else:
                        for (j, value_ij) in enumerate(value_i):
                            if key in __class__.__REAL_LISTS[1]:
                                # 0-based indices i, j to 1-based indices i + 1, j + 1.
                                lines.append(f'  {key}({i + 1},{j + 1}) = {value_ij},')
                            else:
                                for (k, value_ijk) in enumerate(value_ij):
                                  # 0-based indices i, j, k to 1-based indices i + 1, j + 1, k + 1.
                                    lines.append((f'  {key}({i + 1},{j + 1},{k + 1}) = {value_ijk},'))
        lines.append('/')
        return '\n'.join(lines)
    @property
    def lspinorb(self)-> bool:
        '''
        With or without spin-orbit coupling.
        '''
        return self.variables.get('lspinorb', False)
    @property
    def nbnd(self)-> int:
        '''
        Number of bands.
        '''
        return self.variables.get('nbnd')
    @property
    def magmom(self)-> list[float]:
        '''
        Magnetic moments [mu_{B}] of sites.
        '''
        return self.variables.get('starting_magnetization', [0] * self.variables['nat'])
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
        '''
        variables = {}
        while True:
            line = f.readline()
            # Stop when / is found.
            if line.startswith('/'):
                break
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            # Skip blank line.
            if not line:
                continue
            line = line.strip(',')
            # Variables are separated with =.
            line = line.split('=')
            key = line[0].strip()
            value = line[1].strip()
            if key in __class__.__CHARACTERS:
                variables[key] = value.strip("'")
            elif key in __class__.__LOGICALS:
                variables[key] = True if value.strip('.').lower().startswith('t') else False
            elif key in __class__.__INTEGERS:
                variables[key] = int(value)
            elif key in __class__.__REALS:
                variables[key] = float(value)
        return cls(**variables)
    @classmethod
    def from_structure(cls,
                       structure: Structure,
                       lspinorb: bool,
                       nbnd: int = None,
                       occupation: str = 'smearing',
                       degauss: float = 1e-2):
        '''
        Classmethod from Structure object.

        Args:
            structure (Structure):
            lspinorb (bool): With or without spin-orbit coupling.
            nbnd (int, optional): Number of bands. Defaults to None.
            occupation (str, optional): 'smearing', 'tetrahedra', 'tetrahedra_lin', 'tetrahedra_opt', 'fixed', 'from_input'.
                Defaults to 'smearing' (Gaussian).
            degauss (float, optional): Smearing [eV]. Defaults to 1e-2.

        Note:
            Variables: ibrav, nat, ntyp, ecutwfc, occupation, degauss.
            Options: nbnd, starting_magnetization, nspin, noncolin, lspinorb.
        '''
        import json
        # Ordered set of atomic_numbers.
        atomic_numbers = sorted(set(structure.atomic_numbers), key = structure.atomic_numbers.index)
        # Read json file on pseudopotentials.
        textio = open(file = _JSON[lspinorb])
        f = json.load(textio)
        textio.close()
        print(f'## Reading {_JSON[lspinorb]} finished. ##')
        ecutwfc = max(f['pseudos_metadata'][str(Element.from_Z(Z = z))]['hints'][_HINT]['ecut']
                      for z in atomic_numbers)
        options = {}
        if nbnd is not None:
            options['nbnd'] = nbnd
        # If magnetically ordered.
        if sum(map(abs, structure.site_properties['magmom'])) > 0:
            options['starting_magnetization'] = structure.site_properties['magmom']
            if not lspinorb:
                options['nspin'] = 2
        if lspinorb:
            options['noncolin'] = lspinorb
            options['lspinorb'] = lspinorb
        return cls(ibrav = 0, nat = structure.num_sites, ntyp = structure.n_elems, ecutwfc = ecutwfc,
                   occupations = occupation, degauss = degauss / common.CONSTANTS['Ry[eV]'], **options)

class Electrons:
    '''
    Class of &ELECTRONS namelist.

    Args:
        variables (dict[str]): Variables for &ELECTRONS namelist.
    '''
    __CHARACTERS = ['mixing_mode', 'diagonalization', 'efield_phase', 'startingpot', 'startingwfc']
    __LOGICALS = ['scf_must_converge', 'adaptive_thr', 'diago_rmm_conv', 'diago_full_acc', 'tqr',
                  'real_space']
    __INTEGERS = ['electron_maxstep', 'exx_maxstep', 'mixing_ndim', 'mixing_fixed_ns', 'diago_cg_maxiter',
                  'diago_ppcg_maxiter', 'diago_david_ndim', 'diago_rmm_ndim', 'diago_gs_nblock']
    __REALS = ['conv_thr', 'conv_thr_init', 'conv_thr_multi', 'mixing_beta', 'diago_thr_init',
               'efield']
    __REAL_LISTS = [['efield_cart']]
    __NUMBERS = __INTEGERS + __REALS
    def __init__(self,
                 **variables):
        self.variables = variables
    def __str__(self):
        lines = [f'&{__class__.__name__.upper()}']
        for (key, value) in self.variables.items():
            if key in __class__.__CHARACTERS:
                lines.append(f'  {key} = \'{value}\',')
            elif key in __class__.__LOGICALS:
                lines.append(f'  {key} = .{value}.,')
            elif key in __class__.__NUMBERS:
                lines.append(f'  {key} = {value},')
            # Flatten by sum(two_dimensional_list, []).
            elif key in sum(__class__.__REAL_LISTS, []):
                for (i, value_i) in enumerate(value):
                    if key in __class__.__REAL_LISTS[0]:
                        # 0-based index i to 1-based index i + 1.
                        lines.append(f'  {key}({i + 1}) = {value_i},')
        lines.append('/')
        return '\n'.join(lines)
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
        '''
        variables = {}
        while True:
            line = f.readline()
            # Stop when / is found.
            if line.startswith('/'):
                break
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            # Skip blank line.
            if not line:
                continue
            line = line.strip(',')
            # Variables are separated with =.
            line = line.split('=')
            key = line[0].strip()
            value = line[1].strip()
            if key in __class__.__CHARACTERS:
                variables[key] = value.strip("'")
            elif key in __class__.__LOGICALS:
                variables[key] = True if value.strip('.').lower().startswith('t') else False
            elif key in __class__.__INTEGERS:
                variables[key] = int(value)
            elif key in __class__.__REALS:
                variables[key] = float(value)
        return cls(**variables)

class Atomic_Species:
    '''
    Class of ATOMIC_SPECIES card.

    Args:
        elements (list[Element]): Types of elements. Length is ntyp.
        masses (list[float]): Masses of elements in amu unit. Length is ntyp.
        peudopots (list[str]): Names of pseudopotentials. Length is ntyp.
    '''
    def __init__(self,
                 elements: list[Element],
                 masses: list[float],
                 pseudopots: list[str]):
        self.elements = elements
        self.masses = masses
        self.pseudopots = pseudopots
    def __str__(self):
        lines = [f'{__class__.__name__.upper()}']
        for (element, mass, pseudopot) in zip(self.elements, self.masses, self.pseudopots):
            lines.append(f'  {str(element)} {mass} {pseudopot}')
        return '\n'.join(lines)
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper,
                  ntyp: int):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
            ntyp (int): Number of types of atoms.
        '''
        elements = []
        masses = []
        pseudopots = []
        for n in range(ntyp):
            line = f.readline()
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            line = line.split()
            elements.append(Element(line[0]))
            masses.append(float(line[1]))
            pseudopots.append(line[2])
        return cls(elements = elements, masses = masses, pseudopots = pseudopots)
    @classmethod
    def from_structure(cls,
                       structure: Structure):
        '''
        Classmethod from Structure object.

        Args:
            structure (Structure):
        '''
        # Ordered set of atomic_numbers.
        atomic_numbers = sorted(set(structure.atomic_numbers), key = structure.atomic_numbers.index)
        elements = [Element.from_Z(Z = z)
                    for z in atomic_numbers]
        masses = [float(element.atomic_mass)
                  for element in elements]
        pseudopots = [str(element) + '.upf'
                      for element in elements]
        return cls(elements = elements, masses = masses, pseudopots = pseudopots)

class Atomic_Positions:
    '''
    Class of ATOMIC_POSITIONS card.

    Args:
        option (str): alat, bohr, angstrom, crystal, crystal_sg.
        elements (list[Element]): Elements. Length is nat.
        coordinates (np.ndarray[float]): Coordinates. Shape is (nat, 3).
    '''
    def __init__(self,
                 option: str,
                 elements: list[Element],
                 coordinates: np.ndarray[float]):
        self.option = option
        self.elements = elements
        self.coordinates = coordinates
    def __str__(self):
        lines = [f'{__class__.__name__.upper()} {{{self.option}}}']
        for (element, coordinate) in zip(self.elements, self.coordinates):
            (x, y, z) = coordinate
            lines.append(f'  {str(element)} {x} {y} {z}')
        return '\n'.join(lines)
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper,
                  option: str,
                  nat: int):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
            option (str): alat, bohr, angstrom, crystal, crystal_sg.
            nat (int): Number of atoms.
        '''
        elements = []
        coordinates = np.empty(shape = (nat, 3), dtype = float)
        for n in range(nat):
            line = f.readline()
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            line = line.split()
            elements.append(Element(line[0]))
            coordinates[n] = line[1:4]
        return cls(option = option, elements = elements, coordinates = coordinates)
    @classmethod
    def crystal(cls,
                structure: Structure):
        '''
        Classmethod with crystal option from Structure object.

        Args:
            structure (Structure):

        Note:
            coordinates are in fractional coordinates.
        '''
        elements = [Element.from_Z(Z = z)
                    for z in structure.atomic_numbers]
        frac_coords = structure.frac_coords
        return cls(option = 'crystal', elements = elements, coordinates = frac_coords)

class K_Points:
    '''
    Class of K_POINTS card.

    Args:
        option (str): tpiba, automatic, crystal, gamma, tpiba_b, crystal_b, tpiba_c, crystal_c.
        kwargs (dict[str]): Special k-points 'xk', weights 'wk', and 'labels'. Length is nks.
            Kmesh 'nk' and shifts 'sk'. Length is 3.
    '''
    def __init__(self,
                 option: str,
                 **kwargs):
        self.option = option
        self.kwargs = kwargs
    def __str__(self):
        lines = [f'{__class__.__name__.upper()} {{{self.option}}}']
        # For gamma option, write nothing.
        if self.option == 'gamma':
            pass
        # For automatic option, write kmesh and shifts.
        elif self.option == 'automatic':
            (x1, y1, z1) = self.kwargs['nk']
            (x2, y2, z2) = self.kwargs['sk']
            lines.append(f'  {x1} {y1} {z1} {x2} {y2} {z2}')
        else:
            lines.append(f'  {len(self.kwargs.get('xk', []))}')
            if 'labels' in self.kwargs.keys():
                for (kpoint, wk, label) in zip(self.kwargs.get('xk'), self.kwargs.get('wk'), self.kwargs['labels']):
                    (x, y, z) = kpoint
                    lines.append(f'  {x} {y} {z} {wk} ! {label}')
            else:
                for (kpoint, wk) in zip(self.kwargs.get('xk'), self.kwargs.get('wk')):
                    (x, y, z) = kpoint
                    lines.append(f'  {x} {y} {z} {wk}')
        return '\n'.join(lines)
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper,
                  option: str):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
            option (str): tpiba, automatic, crystal, gamma, tpiba_b, crystal_b, tpiba_c, crystal_c.
        '''
        if option == 'automatic':
            line = f.readline()
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            line = line.split()
            nk = [int(value)
                  for value in line[0:3]]
            sk = [int(value)
                  for value in line[3:6]]
            return cls(option = option, nk = nk, sk = sk)
        elif option == 'gamma':
            return cls(option = option)
        else:
            line = f.readline()
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            nks = int(line.strip())
            xk = []
            wk = []
            for k in range(nks):
                line = f.readline()
                # Characters after # and ! are comments.
                line = line.split('#')[0].split('!')[0].strip()
                line = line.split()
                xk.append(np.array(object = line[0:3], dtype = float))
                wk.append(float(line[3]))
            return cls(option = option, xk = xk, wk = wk)
    @classmethod
    def automatic(cls,
                  kmesh: list[int],
                  shifts: list[int]):
        '''
        Classmethod with automatic option.

        Args:
            kmesh (list[int]): Number of k-points for Monkhorst-Pack grids. Length is 3.
            shifts (list[int]): Shifts. Length is 3.
        '''
        # Duplicate kmesh if only one component is specified.
        if isinstance(kmesh, int):
            kmesh = [kmesh]
        if len(kmesh) == 1:
            kmesh = kmesh * 3
        return cls(option = 'automatic', nk = kmesh, sk = shifts)
    @classmethod
    def crystal(cls,
                kmesh: list[int],
                shifts: list[int]):
        '''
        Classmethod with crystal option.

        Args:
            kmesh (list[int]): Number of k-points for Monkhorst-Pack grids. Length is 3.
            shifts (list[int]): Shifts. Length is 3.

        Note:
            kpoints are in fractional coordinates.
        '''
        # Duplicate kmesh if only one component is specified.
        if isinstance(kmesh, int):
            kmesh = [kmesh]
        if len(kmesh) == 1:
            kmesh = kmesh * 3
        kpoints = wannier90.Vectors.monkhorst_pack(kmesh = kmesh, shifts = shifts)
        num_kpoints = kpoints.num_vectors
        return cls(option = 'crystal', xk = kpoints, wk = [1 / num_kpoints] * num_kpoints)
    @classmethod
    def crystal_b(cls,
                  structure: Structure,
                  kmesh: int):
        '''
        Classmethod with crystal_b option from Structure object.

        Args:
            structure (Structure):
            kmesh (int): Number of k-points per line.

        Note:
            kpoints are in fractional coordinates.
        '''
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        # If three components of kmesh are specified, use first.
        if isinstance(kmesh, list):
            kmesh = kmesh[0]
        kpath = HighSymmKpath(structure = structure, has_magmoms = True, magmom_axis = [0, 0, 1]).kpath
        kpoints = []
        weights = []
        labels = []
        for line in kpath['path']:
            for label in line:
                kpoints.append(kpath['kpoints'][label])
                weights.append(kmesh)
                labels.append(label)
            weights[-1] = 1
        return cls(option = 'crystal_b', xk = kpoints, wk = weights, labels = labels)
    @classmethod
    def gamma(cls):
        '''
        Classmethod with gamma option.
        '''
        return cls(option = 'gamma')

class Cell_Parameters:
    '''
    Class of CELL_PARAMETERS card.

    Args:
        option (str): alat, bohr, angstrom.
        avecs (np.ndarray[float]): lattice vectors. Shape is (3, 3).
    '''
    def __init__(self,
                 option: str,
                 avecs: np.ndarray[float]):
        self.option = option
        self.avecs = avecs
    def __str__(self):
        lines = [f'{__class__.__name__.upper()} {{{self.option}}}']
        for avec in self.avecs:
            (x, y, z) = avec
            lines.append(f'  {x} {y} {z}')
        return '\n'.join(lines)
    @classmethod
    def from_file(cls,
                  f: TextIOWrapper,
                  option: str):
        '''
        Classmethod from input file.

        Args:
            f (TextIOWrapper):
            option (str): alat, bohr, angstrom.
        '''
        avecs = np.empty(shape = (3, 3), dtype = float)
        for i in range(3):
            line = f.readline()
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            avecs[i] = line.split()
        return cls(option = option, avecs = avecs)
    @classmethod
    def angstrom(cls,
                 structure: Structure):
        '''
        Classmethod with angstrom option from Structure object.

        Args:
            structure (Structure):

        Note:
            avecs is in angstrom unit.
        '''
        return cls(option = 'angstrom', avecs = structure.lattice.matrix)

class Bands:
    '''
    Class of &BANDS namelist and input file for bands.x.

    Args:
        variables (dict[str]): Variables for &BANDS namelist.
    '''
    __CHARACTERS = ['prefix', 'outdir', 'filband', 'filp']
    __LOGICALS = ['lp', 'lsym', 'no_overlap', 'plot_2d']
    __INTEGERS = ['spin_component', 'firstk', 'lastk']
    __LOGICAL_LISTS = [['lsigma']]
    def __init__(self,
                 **variables):
        self.variables = variables
    def __str__(self):
        lines = [f'&{__class__.__name__.upper()}']
        for (key, value) in self.variables.items():
            if key in __class__.__CHARACTERS:
                lines.append(f'  {key} = \'{value}\',')
            elif key in __class__.__LOGICALS:
                lines.append(f'  {key} = .{value}.,')
            elif key in __class__.__INTEGERS:
                lines.append(f'  {key} = {value},')
            # Flatten by sum(two_dimensional_list, []).
            elif key in sum(__class__.__LOGICAL_LISTS, []):
                for (i, value_i) in enumerate(value):
                    if key in __class__.__LOGICAL_LISTS[0]:
                        # 0-based index i to 1-based index i + 1.
                        lines.append(f'  {key}({i + 1}) = .{value_i}.,')
        lines.append('/')
        return '\n'.join(lines)
    def get_jobscript(self,
                      queue_name: str,
                      num_procs: int,
                      pe_name: str = None)-> common.JobScript:
        '''
        Method to get common.JobScript object for bands.x.

        Args:
            queue_name (str): Queue name.
            num_procs (int): Number of processors.
            pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

        Returns:
            common.Jobscript.
        '''
        job_name = 'bands'
        exe_name = CLUSTER['directories'][_PACKAGE] / 'bands.x'
        postfix = f'< {job_name}.in > {job_name}.out'
        return common.JobScript(job_name = job_name, stdout = 'stdout', queue_name = queue_name,
                                num_procs = num_procs, exe_name = exe_name, postfix = postfix, pe_name = pe_name)
    def write_file(self,
                   qedir: Path):
        '''
        Method to write input file to bands.in.

        Args:
            qedir (Path): Working directory for Quantum Espresso calculations.
        '''
        filename = qedir / 'bands.in'
        with open(file = filename, mode = 'w') as f:
            f.write(str(self))
        print(f'## Saving {filename} finished. ##')

class InputPP:
    '''
    Class of &INPUTPP namelist as well as input file for pw2wannier.x.

    Args:
        variables (dict[str]): Variables for &INPUTPP namelist.
    '''
    __CHARACTERS = ['prefix', 'outdir', 'atom_proj_dir', 'seedname', 'spin_component', 'wan_mode', 'scdm_entanglement']
    __LOGICALS = ['write_unk', 'reduce_unk', 'wvfn_formatted', 'write_amn', 'scdm_proj',
                  'atom_proj', 'atom_proj_ext', 'atom_proj_ortho', 'write_mmn', 'write_spn',
                  'spn_formatted', 'write_uHu', 'uHu_formatted', 'write_uIu', 'uIu_formatted',
                  'write_sHu', 'sHu_formatted', 'write_sIu', 'sIu_formatted', 'write_unkg',
                  'irr_bz', 'write_dmn', 'read_sym']
    __INTEGERS = ['reduce_unk_factor']
    __REALS = ['scdm_mu', 'scdm_sigma']
    __INTEGER_LISTS = ['atom_proj_exclude']
    __NUMBERS = __INTEGERS + __REALS
    def __init__(self,
                 **variables):
        self.variables = variables
    def __str__(self):
        lines = [f'&{__class__.__name__.upper()}']
        for (key, value) in self.variables.items():
            if key in __class__.__CHARACTERS:
                lines.append(f'  {key} = \'{value}\',')
            elif key in __class__.__LOGICALS:
                lines.append(f'  {key} = .{value}.,')
            elif key in __class__.__NUMBERS:
                lines.append(f'  {key} = {value},')
            # Flatten by sum(two_dimensional_list, []).
            elif key in sum(__class__.__INTEGER_LISTS, []):
                for (i, value_i) in enumerate(value):
                    if key in __class__.__INTEGER_LISTS[0]:
                        # 0-based index i to 1-based index i + 1.
                        lines.append(f'  {key}({i + 1}) = {value_i},')
        lines.append('/')
        return '\n'.join(lines)
    def get_jobscript(self,
                      queue_name: str,
                      num_procs: int,
                      pe_name: str = None)-> common.JobScript:
        '''
        Method to get common.JobScript object for pw2wannier90.x.

        Args:
            queue_name (str): Queue name.
            num_procs (int): Number of processors.
            pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

        Returns:
            common.Jobscript.
        '''
        job_name = 'pw2wannier90'
        exe_name = CLUSTER['directories'][_PACKAGE] / 'pw2wannier90.x'
        postfix = f'< {job_name}.in > {job_name}.out'
        return common.JobScript(job_name = job_name, stdout = 'stdout', queue_name = queue_name,
                                num_procs = num_procs, exe_name = exe_name, postfix = postfix, pe_name = pe_name)
    def write_file(self,
                   qedir: Path):
        '''
        Method to write input file to pw2wannier90.in.

        Args:
            qedir (Path): Working directory for Quantum Espresso calculations.
        '''
        filename = qedir / 'pw2wannier90.in'
        with open(file = filename, mode = 'w') as f:
            f.write(str(self))
        print(f'## Saving {filename} finished. ##')

class InputPW:
    '''
    Class of input file for pw.x.

    Args:
        control (Control):
        system (System):
        electrons (Electrons):
        atomic_species (Atomic_Species):
        atomic_positions (Atomic_Positions):
        k_points (K_Points):
        cell_parameters (Cell_Parameters):

    Attributes:
        structure (Structure):
    '''
    def __init__(self,
                 control: Control,
                 system: System,
                 electrons: Electrons,
                 atomic_species: Atomic_Species,
                 atomic_positions: Atomic_Positions,
                 k_points: K_Points,
                 cell_parameters: Cell_Parameters):
        self.control = control
        self.system = system
        self.electrons = electrons
        self.atomic_species = atomic_species
        self.atomic_positions = atomic_positions
        self.k_points = k_points
        self.cell_parameters = cell_parameters
    def __str__(self):
        lines = [str(self.control),
                 str(self.system),
                 str(self.electrons),
                 str(self.atomic_species),
                 str(self.atomic_positions),
                 str(self.k_points),
                 str(self.cell_parameters),
                 '']
        return '\n'.join(lines)
    @property
    def structure(self)-> Structure:
        if self.cell_parameters.option == 'bohr':
            lattice = self.cell_parameters.avec * common.CONSTANTS['Bohr[AA]']
        else:
            lattice = self.cell_parameters.avecs
        species = self.atomic_positions.elements
        if self.atomic_positions.option == 'bohr':
            coords = self.atomic_positions.coordinates * common.CONSTANTS['Bohr[AA]']
            coords_are_cartesian = True
        elif self.atomic_positions.option == 'ang':
            coords = self.atomic_positions.coordinates
            coords_are_cartesian = True
        else:
            coords = self.atomic_positions.coordinates
            coords_are_cartesian = False
        site_properties = {'magmom': self.system.magmom}
        return Structure(lattice = lattice, species = species, coords = coords, coords_are_cartesian = coords_are_cartesian,
                         site_properties = site_properties)
    @classmethod
    def from_file(cls,
                  filename: Path):
        '''
        Classmethod from Structure object.

        Args:
            filename (Path): Name of input file.
        '''
        with open(file = filename) as f:
            while True:
                line = f.readline()
                # Stop at end of file.
                if not line:
                    break
                # Characters after # and ! are comments.
                line = line.split('#')[0].split('!')[0].strip()
                # Skip blank line.
                if not line:
                    continue
                if line.startswith('&'):
                    namelist = line[1:]
                    if namelist == 'CONTROL':
                        control = Control.from_file(f = f)
                    elif namelist == 'SYSTEM':
                        system = System.from_file(f = f)
                    elif namelist == 'ELECTRONS':
                        electrons = Electrons.from_file(f = f)
                else:
                    line = line.split()
                    card = line[0]
                    if len(line) > 1:
                        option = line[1].strip('{').strip('}')
                    if card == 'ATOMIC_SPECIES':
                        atomic_species = Atomic_Species.from_file(f = f, ntyp = system.variables['ntyp'])
                    elif card == 'ATOMIC_POSITIONS':
                        atomic_positions = Atomic_Positions.from_file(f = f, option = option, nat = system.variables['nat'])
                    elif card == 'K_POINTS':
                        k_points = K_Points.from_file(f = f, option = option)
                    elif card == 'CELL_PARAMETERS':
                        cell_parameters = Cell_Parameters.from_file(f = f, option = option)
        print(f'## Reading {filename} finished. ##')
        return cls(control = control, system = system, electrons = electrons, atomic_species = atomic_species,
                   atomic_positions = atomic_positions, k_points = k_points, cell_parameters = cell_parameters)
    @classmethod
    def from_structure(cls,
                       structure: Structure,
                       calculation: str,
                       lspinorb: bool,
                       kmesh: list[int],
                       shifts: list[int] = [0] * 3,
                       nbnd: int = None,
                       occupation: str = 'smearing',
                       degauss: float = 1e-2,
                       conv_thr: float = 1e-5):
        '''
        Classmethod from Structure object.

        Args:
            structure (Structure):
            calculation (str): 'scf', 'nscf', 'bands'. 'relax', 'md', 'vc-relax', 'vc-md' are not acceptable.
            lspinorb (bool): With or without spin-orbit coupling.
            kmesh (list[int]): For bands, number of k-points per line. Otherwise, number of k-points for Monkhorst-Pack grids. Length is 3.
            shifts (list[int], optional): Shifts. Length is 3. Defaults to [0] * 3.
            nbnd (int, optional): Number of bands. Defaults to None.
            occupation (str, optional): 'smearing', 'tetrahedra', 'tetrahedra_lin', 'tetrahedra_opt', 'fixed', 'from_input'.
                Defaults to 'smearing' (Gaussian).
            degauss (float, optional): Smearing [eV]. Defaults to 1e-2.
            conv_thr (float, optional): Convergence criterion [eV]. Defaults to 1e-5.
        '''
        control = Control(calculation = calculation, prefix = structure.reduced_formula,
                          pseudo_dir = _PSEUDO_DIR[lspinorb])
        system = System.from_structure(structure = structure, lspinorb = lspinorb, nbnd = nbnd, occupation = occupation, degauss = degauss)
        electrons = Electrons(conv_thr = conv_thr / common.CONSTANTS['Ry[eV]'])
        atomic_species = Atomic_Species.from_structure(structure = structure)
        atomic_positions = Atomic_Positions.crystal(structure = structure)
        if calculation  == 'scf':
            k_points = K_Points.automatic(kmesh = kmesh, shifts = shifts)
        elif calculation == 'nscf':
            k_points = K_Points.crystal(kmesh = kmesh, shifts = shifts)
        elif calculation == 'bands':
            k_points = K_Points.crystal_b(structure = structure, kmesh = kmesh)
        cell_parameters = Cell_Parameters.angstrom(structure = structure)
        return cls(control = control, system = system, electrons = electrons, atomic_species = atomic_species,
                   atomic_positions = atomic_positions, k_points = k_points, cell_parameters = cell_parameters)
    def get_bands(self)-> Bands:
        '''
        Method to get Bands object.

        Returns:
            Bands:

        Note:
            Variables: prefix, filband, lsym.
        '''
        return Bands(prefix = self.control.variables['prefix'], filband = 'bands', lsym = False)
    def get_inputpp(self,
                    morb: bool = False,
                    shc: bool = False,
                    K_orb: bool = False,
                    K_spin: bool = False,
                    NOA_spin: bool = False,
                    acc_spin: bool = False)-> InputPP:
        '''
        Method to get InputPP object.

        Args:
            morb (bool, optional): Compute morb or not. Defaults to False.
            shc (bool, optional): Compute shc or not. Defaults to False.
            K_orb (bool, optional): Compute K_orb or not. Defaults to False.
            K_spin (bool, optional): Compute K_spin or not. Defaults to False.
            NOA_spin (bool, optional): Compute NOA_spin or not. Defaults to False.
            acc_spin (bool, optional): Compute acc_spin or not. Defaults to False.

        Returns:
            InputPP:

        Note:
            Variables: prefix, seedname.
            Options: write_spn, write_uHu, write_sHu, write_sIu.
        '''
        options = {}
        if morb or K_orb or acc_spin:
            options['write_uHu'] = True
        if shc:
            options['write_sHu'] = True
        if shc or acc_spin:
            options['write_sIu'] = True
        if shc or K_spin or NOA_spin or acc_spin:
            options['write_spn'] = True
        return InputPP(prefix = self.control.variables['prefix'], seedname = 'wannier90', **options)
    def get_jobscript(self,
                      queue_name: str,
                      num_procs: int,
                      pe_name: str = None)-> common.JobScript:
        '''
        Method to get common.JobScript object for pw.x.

        Args:
            queue_name (str): Queue name.
            num_procs (int): Number of processors.
            pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

        Returns:
            common.Jobscript.
        '''
        job_name = f'pw{self.control.variables["calculation"]}'
        exe_name = CLUSTER['directories'][_PACKAGE] / 'pw.x'
        postfix = f'< {job_name}.in > {job_name}.out'
        return common.JobScript(job_name = job_name, stdout = 'stdout', queue_name = queue_name,
                                num_procs = num_procs, exe_name = exe_name, postfix = postfix, pe_name = pe_name)
    def get_win(self,
                projections: wannier90.Projections,
                exclude_bands: wannier90.SpecifyBands = None)-> wannier90.Win:
        '''
        Method to get wannier90.Win object.

        Args:
            projections (wannier90.Projections):
            exclude_bands (wannier90.SpecifyBands, optional): Defaults to None.

        Returns:
            wannier90.Win:

        Note:
            Variables: spinors, num_wann, num_bands, mp_grid, unit_cell_cart, kpoints, atoms_cart, projections.
            Options: exclude_bands.
        '''
        spinors = self.system.lspinorb
        num_wann = projections.num_wann
        if spinors:
            num_wann = 2 * num_wann
        num_exclude = exclude_bands.num_specified if exclude_bands is not None else 0
        num_bands = self.system.nbnd - num_exclude
        unit_cell_cart = wannier90.Vectors('unit_cell_cart', *self.cell_parameters.avecs)
        kvecs = self.k_points.kwargs.get('xk')
        mp_grid = [len(set(kvec[i]
                           for kvec in kvecs))
                   for i in range(3)]
        shifts = [kvecs[0][i] * mp_grid[i]
                  for i in range(3)]
        kpoints = wannier90.Vectors.monkhorst_pack(kmesh = mp_grid, shifts = shifts)
        atoms_cart = wannier90.LabeledVectors(key = 'atoms_cart', labels = self.atomic_positions.elements,
                                              vectors = self.atomic_positions.coordinates)
        system = wannier90.System(spinors = spinors, num_wann = num_wann, num_bands = num_bands, mp_grid = mp_grid,
                                  unit_cell_cart = unit_cell_cart, kpoints = kpoints, atoms_cart = atoms_cart, projections = projections)
        if exclude_bands is None:
            jobcontrol = wannier90.JobControl()
        else:
            jobcontrol = wannier90.JobControl(exclude_bands = exclude_bands)
        return wannier90.Win(system = system, jobcontrol = jobcontrol,
                             disentangle = wannier90.Disentangle(), wannierise = wannier90.Wannierise(), plot = wannier90.Plot(),
                             postw90 = wannier90.Postw90(), berry = wannier90.Berry(), gyrotropic = wannier90.Gyrotropic())
    def write_file(self,
                   qedir: Path):
        '''
        Method to write input file to pwcalculation.in.

        Args:
            qedir (Path): Working directory for Quantum Espresso calculations.
        '''
        filename = qedir / f'pw{self.control.variables["calculation"]}.in'
        with open(file = filename, mode = 'w') as f:
            f.write(str(self))
        print(f'## Saving {filename} finished. ##')

class BandsGnu:
    '''
    Class of output of band structure calculations.

    Args:
        distances (list[float]): Distances [AA^{-1}]. Length is num_kpoints.
        eigvals (np.np.ndarray[float]): Eigenvalues [eV]. Shape is (num_bands, num_kpoints).

    Attributes:
        num_bands (int): Number of bands.
        num_kpoints (int): Number of k-points.
    '''
    def __init__(self,
                 distances,
                 eigvals: np.ndarray[float]):
        self.distances = distances
        self.eigvals = eigvals
    @property
    def num_bands(self)-> int:
        '''
        Number of bands.
        '''
        return len(self.eigvals)
    @property
    def num_kpoints(self)-> int:
        '''
        Number of k-points.
        '''
        return len(self.distances)
    @classmethod
    def from_file(cls,
                  structure: Structure,
                  qedir: Path):
        '''
        Classmethod from output of band structure calculations.

        Args:
            structure (Structure):
            qedir (Path): Working directory for Quantum Espresso calculations.
        '''
        # Distances from dimensionless to [AA^{-1}].
        alat = np.linalg.norm(x = structure.lattice.matrix[0])
        factor = 2 * np.pi / alat
        filename = qedir / 'bands.gnu'
        with open(file = filename) as f:
            eigvals = []
            # Loop over band indices.
            while True:
                line = f.readline()
                line = line.strip()
                # Stop at blank line.
                if not line:
                    break
                distances = []
                # Loop over distances.
                while True:
                    line = line.split()
                    # Format: dimensionless distance, eigval.
                    distances.append(factor * float(line[0]))
                    eigvals.append(line[1])
                    line = f.readline()
                    line = line.strip()
                    # Break at blank line.
                    if not line:
                        break
        print(f'## Reading {filename} finished. ##')
        num_kpoints = len(distances)
        eigvals = np.array(object = eigvals, dtype = float).reshape(-1, num_kpoints)
        return cls(distances = distances, eigvals = eigvals)
    def get_bandplotter(self,
                        structure: Structure,
                        efermi: float)-> common.BandPlotter:
        bandplot = common.BandPlotter.from_structure(structure = structure)
        bandplot.add_bandstructure(bandstructure = self.get_bandstructure(efermi = efermi))
        return bandplot
    def get_bandstructure(self,
                          efermi: float)-> common.BandStructure:
        '''
        Method to get common.BandStructure object.

        Args:
            efermi (float): Pristine Fermi energy [eV].
        '''
        # Subtract pristine Fermi energy.
        return common.BandStructure(distances = self.distances, eigvals = self.eigvals - efermi)

def get_efermi(qedir: Path)-> float:
    '''
    Method to get Fermi energy.

    Args:
        qedir (Path): Working directory for Quantum Espresso calculations.

    Returns:
        float: Fermi energy [eV].
    '''
    filename = qedir / 'pwscf.out'
    with open(file = filename) as f:
        while True:
            line = f.readline()
            # Stop at end of file.
            if not line:
                break
            line = line.strip()
            # Format: the Fermi energy is efermi ev.
            if line.startswith('the Fermi energy is'):
                line = line.split()
                efermi = float(line[-2])
    print(f'## Reading {filename} finished. ##')
    return efermi
