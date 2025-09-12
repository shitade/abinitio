'''
Module to deal with Wannier90 input/output files.
'''

from io import TextIOWrapper
import numpy as np
from pathlib import Path
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from abinitio import CLUSTER, common

_PACKAGE = 'wannier90'

class Eig:
    '''
    Class of seedname.eig.

    Args:
        eigvals (np.ndarray[float]): Eigenvalues [eV]. Shape is (num_bands, num_kpoints).

    Attributes:
        num_bands (int): Number of bands.
        num_kpoints (int): Number of k-points.
    '''
    def __init__(self,
                 eigvals: np.ndarray[float]):
        self.eigvals = eigvals
    @property
    def num_bands(self)-> int:
        '''
        Number of bands.
        '''
        return self.eigvals.shape[0]
    @property
    def num_kpoints(self)-> int:
        '''
        Number of k-points
        '''
        return self.eigvals.shape[1]
    @classmethod
    def from_file(cls,
                  wanndir: Path,
                  seedname: str):
        '''
        Classmethod from seedname.eig.

        Args:
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str): Seedname.
        '''
        filename = wanndir / f'{seedname}.eig'
        with open(file = filename) as f:
            ns = []
            ks = []
            eigvals = []
            while True:
                line = f.readline()
                # Stop at end of file.
                if not line:
                    break
                line = line.strip().split()
                # Format: n, k, eigval.
                ns.append(int(line[0]))
                ks.append(int(line[1]))
                eigvals.append(float(line[2]))
        print(f'## Reading {filename} finished. ##')
        num_bands = len(set(ns))
        num_kpoints = len(set(ks))
        eigvals = np.array(object = eigvals).reshape(num_kpoints, num_bands).transpose()
        return cls(eigvals = eigvals)

class Projections:
    '''
    Class of projections block of system parameters.

    Args:
        sites (list[str]): Atomic and nonatomic sites. Length is num_sites.
        orbitals (list[list[str]]): Initial orbitals of sites. Length is num_sites..
        degeneracies (list[int]): Degeneracies of sites. Length is num_sites.

    Attributes:
        num_sites (int): Number of sites.
        num_wann (int): Number of Wannier orbitals without spin-orbit coupling.
    '''
    # Use __DEGENERACIES.get('s', 1) for nondegenerate orbitals such as 's'.
    __DEGENERACIES = {
        'p': 3, 'd': 5, 'f': 7, 'sp': 2, 'sp2': 3, 'sp3': 4, 'sp3d': 5, 'sp3d2': 6
    }
    def __init__(self,
                 sites: list[str],
                 orbitals: list[list[str]],
                 degeneracies: list[int]):
        self.sites = sites
        self.orbitals = orbitals
        self.degeneracies = degeneracies
    def __str__(self):
        lines = [f'begin {__class__.__name__.lower()}']
        for (site, orbital) in zip(self.sites, self.orbitals):
            lines.append(f'  {site}:{";".join(orbital)}')
        lines.append(f'end {__class__.__name__.lower()}')
        return '\n'.join(lines)
    @property
    def nsites(self)-> int:
        '''
        Number of sites.
        '''
        return len(self.sites)
    @property
    def num_wann(self)-> int:
        '''
        Number of Wannier orbitals without spin-orbit coupling.
        '''
        return sum(sum(__class__.__DEGENERACIES.get(wave, 1)
                       for wave in orbital) * degeneracy
                   for (orbital, degeneracy) in zip(self.orbitals, self.degeneracies))
    @classmethod
    def from_structure(cls,
                       structure: Structure,
                       *atomic_orbitals: list[str],
                       **nonatomic_orbitals: list[str]):
        '''
        Classmethod from Structure object, etc.

        Args:
            structure (Structure):
            *atomic_orbitals (tuple[list[str]]): Initial orbitals of atomic sites.
            **nonatomic_orbitals (dict[str, list[str]]): Nonatomic sites and their initial orbitals.
        '''
        # Ordered set of atomic_numbers.
        atomic_numbers = sorted(set(structure.atomic_numbers), key = structure.atomic_numbers.index)
        sites = [str(Element.from_Z(Z = z))
                 for z in atomic_numbers] + list(nonatomic_orbitals)
        orbitals = list(atomic_orbitals) + list(nonatomic_orbitals.values())
        degeneracies = [structure.atomic_numbers.count(z)
                        for z in atomic_numbers] + [1] * len(nonatomic_orbitals)
        return cls(sites = sites, orbitals = orbitals, degeneracies = degeneracies)
    @classmethod
    def from_block(cls,
                   block: list[str]):
        '''
        Classmethod from block in seedname.win.

        Args:
            block (list[str]): Block in seedname.win.
        '''
        sites = []
        orbitals = []
        for line in block:
            line = line.split(':')
            sites.append(line[0])
            orbitals.append(line[1].split(';'))
        return cls(sites = sites, orbitals = orbitals, degeneracies = None)

class Vectors(list):
    '''
    Class of unit_cell_cart and kpoints blocks of system parameters and explicit_kpath block of plot parameters.

    Args:
        key (str): 'unit_cell_cart', 'kpoints', 'explicit_kpath'.
        *vectors (list[np.ndarray[float]]): Vectors. Each shape is (3,).

    Attributes:
        num_vectors (int): Number of vectors.
    '''
    def __init__(self,
                 key: str,
                 *vectors: list[np.ndarray[float]]):
        self.key = key
        super().__init__(vectors)
    def __str__(self):
        lines = [f'begin {self.key.lower()}']
        for vector in self:
            (x, y, z) = vector
            lines.append(f'  {x} {y} {z}')
        lines.append(f'end {self.key.lower()}')
        return '\n'.join(lines)
    @property
    def num_vectors(self)-> int:
        '''
        Number of vectors.
        '''
        return len(self)
    @classmethod
    def from_block(cls,
                   key: str,
                   block: list[str]):
        '''
        Classmethod from block in seedname.win.

        Args:
            key (str): 'unit_cell_cart', 'kpoints', 'explicit_kpath'.
            block (list[str]): Block in seedname.win.
        '''
        factor = 1
        vectors = []
        for line in block:
            line = line.split()
            if len(line) == 1 and line[0].lower() == 'bohr':
                factor = common.CONSTANTS['Bohr[AA]']
            else:
                vectors.append(factor * np.array(object = line, dtype = float))
        return cls(key, *vectors)
    @classmethod
    def monkhorst_pack(cls,
                       kmesh: list[int],
                       shifts: list[int]):
        '''
        Classmethod of Monkhorst-Pack grid.

        Args:
            kmesh (list[int]): Number of k-points for Monkhorst-Pack grids. Length is 3.
            shifts (list[int]): Shifts. Length is 3.
        '''
        kvecs = [np.array(object = [(n0 + shifts[0]) / kmesh[0],
                                    (n1 + shifts[1]) / kmesh[1],
                                    (n2 + shifts[2]) / kmesh[2]], dtype = float)
                 for n0 in range(kmesh[0])
                 for n1 in range(kmesh[1])
                 for n2 in range(kmesh[2])]
        return cls('kpoints', *kvecs)

class LabeledVectors:
    '''
    Class of atoms_cart and atoms_frac of system parameters and explicit_kpath_labels of plot parameters.

    Args:
        key (str): 'atoms_cart', 'atoms_frac', 'explicit_kpath_labels'.
        labels (list[str]): Labels of vectors. Length is num_vectors.
        vectors (list[np.ndarray[float]]): Vectors. Length is num_vectors.

    Attributes:
        num_vectors (int): Number of vectors.
    '''
    def __init__(self,
                 key: str,
                 labels: list[str],
                 vectors: list[np.ndarray[float]]):
        self.key = key
        self.labels = labels
        self.vectors = vectors
    def __str__(self):
        lines = [f'begin {self.key.lower()}']
        for (label, vector) in zip(self.labels, self.vectors):
            (x, y, z) = vector
            lines.append(f'  {label} {x} {y} {z}')
        lines.append(f'end {self.key.lower()}')
        return '\n'.join(lines)
    @property
    def num_vectors(self)-> int:
        '''
        Number of vectors.
        '''
        return len(self.labels)
    @classmethod
    def from_block(cls,
                   key: str,
                   block: list[str]):
        '''
        Classmethod from block in seedname.win.

        Args:
            key (str): 'atoms_cart', 'atoms_frac', 'explicit_kpath_labels'
            block (list[str]): Block in seedname.win.
        '''
        factor = 1
        labels = []
        vectors = []
        for line in block:
            line = line.split()
            if len(line) == 1 and line[0].lower() == 'bohr':
                factor = common.CONSTANTS['Bohr[AA]']
            else:
                labels.append(str(line[0]))
                vectors.append(factor * np.array(object = line[1:4], dtype = float))
        return cls(key = key, labels = labels, vectors = vectors)

class SpecifyBands(set):
    '''
    Class of exclude_bands and select_projections parameters of job control parameters
        and wannier_plot_list and bands_plot_project parameters of plot parameters.

    Args:
        key (str): 'exclude_bands', 'select_projections', 'wannier_plot_list', 'bands_plot_project'.
        *bands (int): Band indices.

    Attributes:
        num_specified (int): Number of specified bands.
    '''
    def __init__(self,
                 key: str,
                 *bands: int):
        self.key = key
        super().__init__(bands)
    def __str__(self):
        return f'{self.key.lower()} = {", ".join(str(index) for index in self)}'
    @property
    def num_specified(self)-> int:
        '''
        Number of specified bands.
        '''
        return len(self)
    @classmethod
    def from_string(cls,
                    key: str,
                    string: str):
        '''
        Classmethod from string in seedname.win.

        Args:
            key (str): 'exclude_bands', 'select_projections', 'wannier_plot_list', 'bands_plot_project'.
            string (str): String in seedname.win.
        '''
        # Bands are separated by ,.
        string = string.split(',')
        # m-n indicates m, m + 1, ..., n.
        string = [value.strip().split('-')
                  for value in string]
        string = [[int(value[0])] if len(value) == 1 else list(range(int(value[0]), int(value[1]) + 1))
                  for value in string]
        # Flatten by sum(two_dimensional_list, []).
        return cls(key, *sum(string, []))

class Kpoint_Path:
    '''
    Class of kpoint_path block of plot parameters.

    Args:
        labels_start (list[str]): Labels of starts of high-symmetric lines. Length is num_lines.
        kpoints_start (list[np.ndarray[float]]): Fractional k-points of ends of high-symmetric lines. Length is num_lines.
        labels_end (list[str]): Labels of ends of high-symmetric lines. Length is num_lines.
        kpoints_end (list[np.ndarray[float]]): Fractional k-points of ends of high-symmetric lines. Length is num_lines.

    Attributes:
        num_lines (int): Number of high-symmetric lines.
    '''
    def __init__(self,
                 labels_start: list[str],
                 kpoints_start: list[np.ndarray[float]],
                 labels_end: list[str],
                 kpoints_end: list[np.ndarray[float]]):
        self.labels_start = labels_start
        self.kpoints_start = kpoints_start
        self.labels_end = labels_end
        self.kpoints_end = kpoints_end
    def __str__(self):
        lines = [f'begin {__class__.__name__.lower()}']
        for (label_start, kpoint_start, label_end, kpoint_end) in zip(self.labels_start, self.kpoints_start, self.labels_end, self.kpoints_end):
            (x1, y1, z1) = kpoint_start
            (x2, y2, z2) = kpoint_end
            lines.append(f'  {label_start} {x1} {y1} {z1} {label_end} {x2} {y2} {z2}')
        lines.append(f'end {__class__.__name__.lower()}')
        return '\n'.join(lines)
    @property
    def num_lines(self)-> int:
        '''
        Number of high-symmetric lines.
        '''
        return len(self.labels_start)
    @classmethod
    def from_structure(cls,
                       structure: Structure):
        '''
        Classmethod from Structure object, etc.

        Args:
            structure (Structure):
        '''
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        kpath = HighSymmKpath(structure = structure, has_magmoms = True, magmom_axis = [0, 0, 1]).kpath
        labels_start = []
        kpoints_start = []
        labels_end = []
        kpoints_end = []
        for line in kpath['path']:
            for (i, label) in enumerate(line):
                if i < len(line) - 1:
                    labels_start.append(label)
                    kpoints_start.append(kpath['kpoints'][label])
                if i > 0:
                    labels_end.append(label)
                    kpoints_end.append(kpath['kpoints'][label])
        return cls(labels_start = labels_start, kpoints_start = kpoints_start, labels_end = labels_end, kpoints_end = kpoints_end)
    @classmethod
    def from_block(cls,
                   block: list[str]):
        '''
        Classmethod from block in seedname.win.

        Args:
            block (list[str]): Block in seedname.win.
        '''
        labels_start = []
        kpoints_start = []
        labels_end = []
        kpoints_end = []
        for line in block:
            line = line.split()
            labels_start.append(str(line[0]))
            kpoints_start.append(np.array(object = line[1:4]))
            labels_end.append(str(line[4]))
            kpoints_end.append(np.array(object = line[5:8]))
        return cls(labels_start = labels_start, kpoints_start = kpoints_start, labels_end = labels_end, kpoints_end = kpoints_end)

class Berry_Task(set):
    '''
    Class of berry_task parameters of berry parameters.

    Args:
        *tasks (set[str]): berry_task.
    '''
    def __init__(self,
                 *tasks: set[str]):
        super().__init__(tasks)
    def __str__(self):
        return f'{__class__.__name__.lower()} = {", ".join(self)}'
    @classmethod
    def from_string(cls,
                    string: str):
        '''
        Classmethod from string in seedname.win.

        Args:
            string (str): String in seedname.win.
        '''
        # Tasks are separated by ,.
        string = string.split(',')
        return cls(*string)

class Gyrotropic_Task(set):
    '''
    Class of gyrotropic_task parameters of gyrotropic parameters.

    Args:
        *task (set[str]): gyrotropic_task.
    '''
    def __init__(self,
                 *tasks: set[str]):
        super().__init__(tasks)
    def __str__(self):
        return f'{__class__.__name__.lower()} = {"-" + "-".join(self)}'
    @classmethod
    def from_string(cls,
                    string: str):
        '''
        Classmethod from string in seedname.win.

        Args:
            string (str): String in seedname.win.
        '''
        # Tasks are separated by -.
        string = string.split('-')
        return cls(*string)

class System:
    '''
    Class of system parameters.

    Args:
        **params (dict[str]): System parameters.

    Attributes:
        structure (Structure):

    Note:
        shell_list parameter and nnkpts block are not implemented.
    '''
    __LOGICALS = ['gamma_only', 'spinors', 'skip_b1_tests', 'higher_order_nearest_shells']
    __INTEGERS = ['num_wann', 'num_bands', 'search_shells', 'higher_order_n']
    __INTEGER_LISTS = ['mp_grid']
    __REALS = ['kmesh_tol']
    __VECTORS = ['unit_cell_cart', 'kpoints']
    __LABELED_VECTORS = ['atoms_cart', 'atoms_frac']
    __PROJECTIONS = ['projections']
    __NUMBERS = __INTEGERS + __REALS
    __BLOCKS = __VECTORS + __LABELED_VECTORS + __PROJECTIONS
    PARAMS = __LOGICALS + __NUMBERS + __INTEGER_LISTS + __BLOCKS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
            elif key in __class__.__INTEGER_LISTS:
                (x, y, z)= value
                lines.append(f'{key} = {x} {y} {z}')
            elif key in __class__.__BLOCKS:
                lines.append(str(value))
        lines.append('')
        return '\n'.join(lines)
    @property
    def structure(self):
        '''
        Structure object.
        '''
        if 'atoms_cart' in self.params.keys():
            atoms_cart = self.params['atoms_cart']
            return Structure(lattice = self.params['unit_cell_cart'], species = atoms_cart.labels, coords = atoms_cart.vectors,
                             coords_are_cartesian = True)
        elif 'atoms_frac' in self.params.keys():
            atoms_frac = self.params['atoms_frac']
            return Structure(lattice = self.params['unit_cell_cart'], species = atoms_frac.labels, coords = atoms_frac.vectors,
                             coords_are_cartesian = False)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of system parameters.
            value (str or list[str]): String or block from seedname.win.
        '''
        if key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__INTEGER_LISTS:
            self.params[key] = [int(string)
                                for string in value.split()]
        elif key in __class__.__REALS:
            self.params[key] = float(value)
        elif key in __class__.__VECTORS:
            self.params[key] = Vectors.from_block(key = key, block = value)
        elif key in __class__.__LABELED_VECTORS:
            self.params[key] = LabeledVectors.from_block(key = key, block = value)
        elif key in __class__.__PROJECTIONS:
            self.params[key] = Projections.from_block(block = value)

class JobControl:
    '''
    Class of job control parameters.

    Args:
        **params (dict[str]): Job control parameters.
    '''
    __CHARACTERS = ['restart', 'length_unit', 'spin', 'devel_flag']
    __LOGICALS = ['postproc_setup', 'auto_projections', 'wvfn_formatted', 'translate_home_cell', 'write_xyz',
                  'write_vdw_data', 'write_hr_diag']
    __INTEGERS = ['iprint', 'timing_level', 'optimisation']
    __SPECIFY_BANDS = ['exclude_bands', 'select_projections']
    __NUMBERS = __CHARACTERS + __INTEGERS
    PARAMS = __LOGICALS + __NUMBERS + __SPECIFY_BANDS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
            elif key in __class__.__SPECIFY_BANDS:
                lines.append(str(value))
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of job control parameters.
            value (str): String from seedname.win.
        '''
        if key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__SPECIFY_BANDS:
            self.params[key] = SpecifyBands.from_string(key = key, string = value)

class Disentangle:
    '''
    Class of disentangle parameters.

    Args:
        **params (dict[str]): Disentangle parameters.

    Note:
        dis_spheres block is not implemented.
    '''
    __LOGICALS = ['dis_froz_proj']
    __INTEGERS = ['dis_num_iter', 'dis_conv_window', 'dis_spheres_num', 'dis_spheres_first_wann']
    __REALS = ['dis_win_min', 'dis_win_max', 'dis_froz_min', 'dis_froz_max', 'dis_proj_min',
               'dis_proj_max', 'dis_mix_ratio', 'dis_conv_tol']
    __NUMBERS = __INTEGERS + __REALS
    PARAMS = __LOGICALS + __NUMBERS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key == 'comment':
                lines.append(f'# {value}.')
            elif key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of disentangle parameters.
            value (str): String from seedname.win.
        '''
        if key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__REALS:
            self.params[key] = float(value)
    @classmethod
    def from_eig(cls,
                 eig: Eig,
                 num_wann: int,
                 efermi: float,
                 dis_num_iter: int = 10000,
                 dis_mix_ratio: float = 0.8,
                 dis_win_max: float = None,
                 dis_froz_max: float = None):
        '''
        Classmethod.

        Args:
            eig (Eig): From seedname.eig.
            num_wann (int): Number of Wannier orbitals.
            efermi (float): Pristine Fermi energy [eV].
            dis_num_iter (int, optional): Number of iterations for minimization. Defaults to 10000.
            dis_mix_ratio (float, optional): Mixing ratio during minimization. Defaults to 0.8.
            dis_win_max (float, optional): Upper bound of outer window [eV] measured from pristine Fermi energy. Defaults to None.
            dis_froz_max (float, optional): Upper bound of inner window [eV] measured from pristine Fermi energy. Defaults to None.

        Note:
            Parameters: dis_num_iter, dis_mix_ratio.
            Options: dis_win_max, dis_froz_max.
        '''
        # Lower bound of dis_win_max = max_{k}[\epsilon_{num_wann - 1}^{(b)}(k)],
        # namely, dis_win_max should be larger than this value.
        lower_bound_dis_win_max = np.amax(a = eig.eigvals[num_wann - 1])
        # Upper bound of dis_froz_max = min_{k}[\epsilon_{num_wann}^{(b)}(k)],
        # namely, dis_froz_max should be smaller than this value.
        upper_bound_dis_froz_max = np.amin(a = eig.eigvals[num_wann])
        comment = f'dis_win_max > {lower_bound_dis_win_max}, dis_froz_max < {upper_bound_dis_froz_max}, efermi = {efermi}'
        options = {}
        if dis_win_max is not None:
            options['dis_win_max'] = max(dis_win_max + efermi, lower_bound_dis_win_max)
        if dis_froz_max is not None:
            options['dis_froz_max'] = min(dis_froz_max + efermi, upper_bound_dis_froz_max)
        return cls(comment = comment, dis_num_iter = dis_num_iter, dis_mix_ratio = dis_mix_ratio, **options)

class Wannierise:
    '''
    Class of wannierise parameters.

    Args:
        **params (dict[str]): Wannierise parameters.

    Note:
        slwf_centres block is not implemented.
    '''
    __LOGICALS = ['precond', 'write_r2mn', 'guiding_centres', 'use_bloch_phases', 'site_symmetry',
                  'slwf_constrain', 'use_ss_functional']
    __INTEGERS = ['num_iter', 'num_cg_steps', 'conv_window', 'conv_noise_num', 'num_dump_cycles',
                  'num_print_cycles', 'num_guide_cycles', 'num_no_guide_iter', 'slwf_num']
    __REALS = ['conv_tol', 'conv_noise_amp', 'trial_step', 'fixed_step', 'symmetrize_eps',
               'slwf_lambda']
    __NUMBERS = __INTEGERS + __REALS
    PARAMS = __LOGICALS + __NUMBERS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of wannierise parameters.
            value (str): String from seedname.win.
        '''
        if key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__REALS:
            self.params[key] = float(value)

class Plot:
    '''
    Class of plot parameters.

    Args:
        **params (dict[str]): Plot parameters.
    '''
    __CHARACTERS = ['wannier_plot_format', 'wannier_plot_mode', 'wannier_plot_spinor_mode', 'bands_plot_format', 'bands_plot_mode',
                    'fermi_surface_plot_format', 'dist_cutoff_mode']
    __LOGICALS = ['wannier_plot', 'wannier_plot_spinor_phase', 'bands_plot', 'fermi_surface_plot', 'write_hr',
                  'write_rmn', 'write_bvec', 'write_tb', 'use_ws_distance', 'write_u_matrices']
    __INTEGERS = ['wannier_plot_supercell', 'bands_num_points', 'bands_plot_dim', 'fermi_surface_num_points', 'ws_search_size']
    __REALS = ['wannier_plot_radius', 'wannier_plot_scale', 'fermi_energy', 'fermi_energy_min', 'fermi_energy_max',
               'fermi_energy_step', 'hr_cutoff', 'dist_cutoff', 'ws_distance_tol']
    __REAL_LISTS = ['translation_centre_frac']
    __VECTORS = ['explicit_kpath']
    __LABELED_VECTORS = ['explicit_kpath_labels']
    __SPECIFY_BANDS = ['wannier_plot_list', 'bands_plot_project']
    __KPOINT_PATH = ['kpoint_path']
    __NUMBERS = __CHARACTERS + __INTEGERS + __REALS
    __BLOCKS = __VECTORS + __LABELED_VECTORS + __SPECIFY_BANDS + __KPOINT_PATH
    PARAMS = __LOGICALS + __NUMBERS + __REAL_LISTS + __BLOCKS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
            elif key in __class__.__REAL_LISTS:
                (x, y, z)= value
                lines.append(f'{key} = {x} {y} {z}')
            elif key in __class__.__BLOCKS:
                lines.append(str(value))
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of plot parameters.
            value (str or list[str]): String or block from seedname.win.
        '''
        if key in __class__.__CHARACTERS:
            self.params[key] = str(value)
        elif key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__REALS:
            self.params[key] = float(value)
        elif key in __class__.__REAL_LISTS:
            self.params[key] = [float(string)
                                for string in value.split()]
        elif key in __class__.__VECTORS:
            self.params[key] = Vectors.from_block(key = key, block = value)
        elif key in __class__.__LABELED_VECTORS:
            self.params[key] = LabeledVectors.from_block(key = key, block = value)
        elif key in __class__.__SPECIFY_BANDS:
            self.params[key] = SpecifyBands.from_string(key = key, string = value)
        elif key in __class__.__KPOINT_PATH:
            self.params[key] = Kpoint_Path.from_block(block = value)
    @classmethod
    def from_structure(cls,
                       structure: Structure,
                       bands_num_points: int):
        '''
        Classmethod from Structure object.

        Args:
            structure (Structure):
            bands_num_points (int): Number of k-points per line.

        Note:
            Parameters: bands_plot, kpoint_path, bands_num_points
            Options: fermi_energy_min, fermi_energy_max.
        '''
        kpoint_path = Kpoint_Path.from_structure(structure = structure)
        return cls(bands_plot = True, kpoint_path = kpoint_path, bands_num_points = bands_num_points)

class Postw90:
    '''
    Class of global parameters of postw90.

    Args:
        **params (dict[str]): Postw90 parameters.
    '''
    __CHARACTERS = ['smr_type', 'berry_curv_unit']
    __LOGICALS = ['adpt_smr', 'spin_decomp', 'spin_moment', 'uHu_formatted', 'spn_formatted',
                  'transl_inv', 'transl_inv_full']
    __INTEGERS = ['num_elec_per_state', 'num_valence_bands']
    __INTEGER_LISTS = ['kmesh']
    __REALS = ['kmesh_spacing', 'adpt_smr_fac', 'adpt_smr_max', 'smr_fixed_en_width', 'scissors_shift',
               'spin_axis_polar', 'spin_axis_azimuth']
    __NUMBERS = __CHARACTERS + __INTEGERS + __REALS
    PARAMS = __LOGICALS + __NUMBERS + __INTEGER_LISTS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
            elif key in __class__.__INTEGER_LISTS:
                (x, y, z)= value
                lines.append(f'{key} = {x} {y} {z}')
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of plot parameters.
            value (str): String from seedname.win.
        '''
        if key in __class__.__CHARACTERS:
            self.params[key] = str(value)
        elif key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__INTEGER_LISTS:
            self.params[key] = [int(string)
                                for string in value.split()]
        elif key in __class__.__REALS:
            self.params[key] = float(value)

class Berry:
    '''
    Class of berry parameters.

    Args:
        **params (dict[str]): Berry parameters.
    '''
    __CHARACTERS = ['kubo_smr_type', 'shc_method']
    __LOGICALS = ['berry', 'kubo_adpt_smr', 'sc_use_eta_corr', 'shc_freq_scan', 'shc_bandshift']
    __INTEGERS = ['berry_curv_adpt_kmesh', 'sc_phase_conv', 'shc_alpha', 'shc_beta', 'shc_gamma',
                  'shc_bandshift_firstband', 'kdotp_num_bands']
    __INTEGER_LISTS = ['berry_kmesh']
    __REALS = ['berry_kmesh_spacing', 'berry_curv_adpt_kmesh_thresh', 'kubo_freq_min', 'kubo_freq_max', 'kubo_freq_step',
               'kubo_eigval_max', 'kubo_adpt_smr_fac', 'kubo_adpt_smr_max', 'kubo_smr_fixed_en_width', 'sc_eta',
               'sc_w_thr', 'shc_bandshift_energyshift']
    __REAL_LISTS = ['kdotp_kpoint']
    __SPECIFY_BANDS = ['kdotp_bands']
    __BERRY_TASK = ['berry_task']
    __NUMBERS = __CHARACTERS + __INTEGERS + __REALS
    __LISTS = __INTEGER_LISTS + __REAL_LISTS
    __BLOCKS = __SPECIFY_BANDS + __BERRY_TASK
    PARAMS = __LOGICALS + __NUMBERS + __LISTS + __BLOCKS
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
            elif key in __class__.__LISTS:
                (x, y, z)= value
                lines.append(f'{key} = {x} {y} {z}')
            elif key in __class__.__BLOCKS:
                lines.append(str(value))
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of plot parameters.
            value (str): String from seedname.win.
        '''
        if key in __class__.__CHARACTERS:
            self.params[key] = str(value)
        elif key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGERS:
            self.params[key] = int(value)
        elif key in __class__.__INTEGER_LISTS:
            self.params[key] = [int(string)
                                for string in value.split()]
        elif key in __class__.__REALS:
            self.params[key] = float(value)
        elif key in __class__.__REAL_LISTS:
            self.params[key] = [float(string)
                                for string in value.split()]
        elif key in __class__.__SPECIFY_BANDS:
            self.params[key] = SpecifyBands.from_string(key = key, string = value)
        elif key in __class__.__BERRY_TASK:
            self.params[key] = Berry_Task.from_string(string = value)
    @classmethod
    def spin_hall(cls,
                  berry: bool,
                  berry_kmesh: list[int],
                  component: common.Component,
                  shc_method: str = 'ryoo'):
        '''
        Classmethod for berry_task = shc to compute spin Hall conductivity.

        Args:
            berry (bool): Use berry module or not.
            berry_kmesh (list[int]): Number of k-points for berry calculations. Length is 3.
            component (common.Component): Of spin Hall conductivity.
            shc_method (str, optional). 'qiao' or 'ryoo'. Defaults to 'ryoo'.
        '''
        # Duplicate berry_kmesh if only one component is specified.
        if isinstance(berry_kmesh, int):
            berry_kmesh = [berry_kmesh]
        if len(berry_kmesh) == 1:
            berry_kmesh = berry_kmesh * 3
        berry_task = Berry_Task('shc')
        to_tuple = component.to_tuple
        # 0-based index to 1-based index.
        return cls(berry = berry, berry_task = berry_task, berry_kmesh = berry_kmesh, shc_method = shc_method,
                   shc_gamma = to_tuple[0] + 1, shc_alpha = to_tuple[1] + 1, shc_beta = to_tuple[2] + 1)

class Gyrotropic:
    '''
    Class of gyrotropic parameters.

    Args:
        **params (dict[str]): Gyrotropic parameters.
    '''
    __CHARACTERS = ['gyrotropic_smr_type']
    __LOGICALS = ['gyrotropic']
    __INTEGER_LISTS = ['gyrotropic_kmesh']
    __REALS = ['gyrotropic_kmesh_spacing', 'gyrotropic_freq_min', 'gyrotropic_freq_max', 'gyrotropic_freq_step', 'gyrotropic_eigval_max',
               'gyrotropic_degen_thresh', 'gyrotropic_smr_fixed_en_width']
    __REAL_LISTS = ['gyrotropic_box_center', 'gyrotropic_box_b1', 'gyrotropic_box_b2', 'gyrotropic_box_b3']
    __GYROTROPIC_TASK = ['gyrotropic_task']
    __NUMBERS = __CHARACTERS + __REALS
    __LISTS = __INTEGER_LISTS + __REAL_LISTS
    PARAMS = __LOGICALS + __NUMBERS + __LISTS + __GYROTROPIC_TASK
    def __init__(self,
                 **params):
        self.params = params
    def __str__(self):
        lines = [f'##~~~~~~~ {__class__.__name__} parameters ~~~~~~~##']
        for (key, value) in self.params.items():
            if key in __class__.__LOGICALS:
                lines.append(f'{key} = .{value}.')
            elif key in __class__.__NUMBERS:
                lines.append(f'{key} = {value}')
            elif key in __class__.__LISTS:
                (x, y, z) = value
                lines.append(f'{key} = {x} {y} {z}')
            elif key in __class__.__GYROTROPIC_TASK:
                lines.append(str(value))
        lines.append('')
        return '\n'.join(lines)
    def add_param(self,
                  key: str,
                  value: str):
        '''
        Method to add parameter.

        Args:
            key (str): Key of plot parameters.
            value (str): String from seedname.win.
        '''
        if key in __class__.__CHARACTERS:
            self.params[key] = str(value)
        elif key in __class__.__LOGICALS:
            self.params[key] = True if value.strip('.').lower().startswith('t') else False
        elif key in __class__.__INTEGER_LISTS:
            self.params[key] = [int(string)
                                for string in value.split()]
        elif key in __class__.__REALS:
            self.params[key] = float(value)
        elif key in __class__.__REAL_LISTS:
            self.params[key] = [float(string)
                                for string in value.split()]
        elif key in __class__.__GYROTROPIC_TASK:
            self.params[key] = Gyrotropic_Task.from_string(string = value)
    @classmethod
    def spin_hall(cls,
                  gyrotropic: bool,
                  gyrotropic_kmesh: list[int],
                  gyrotropic_smr_type: str = 'f-d',
                  gyrotropic_smr_fixed_en_width: float = 300):
        '''
        Classmethod for gyrotropic_task = -spin-acc(-K) to compute spin accumulation (and Edelstein) coefficient.

        Args:
            gyrotropic (bool): Use gyrotropic module or not.
            gyrotropic_kmesh (list[int]): Number of k-points for gyrotropic calculations. Length is 3.
            gyrotropic_smr_type (str): (gauss), 'm-pN' (N >= 0), 'm-v' or 'cold', 'f-d'. Defaults to 'f-d' (Derivative of Fermi distribution function).
            gyrotropic_smr_fixed_en_width (float, optional): Smearing [K] for gyrotropic calculations. Defaults to 300.

        Note:
            If not (edelstein or accumulation), return Error.
        '''
        # Duplicate gyrotropic_kmesh if only one component is specified.
        if isinstance(gyrotropic_kmesh, int):
            gyrotropic_kmesh = [gyrotropic_kmesh]
        if len(gyrotropic_kmesh) == 1:
            gyrotropic_kmesh = gyrotropic_kmesh * 3
        gyrotropic_task = Gyrotropic_Task('acc', 'spin')
        return cls(gyrotropic = gyrotropic, gyrotropic_task = gyrotropic_task, gyrotropic_kmesh = gyrotropic_kmesh,
                   gyrotropic_smr_type = gyrotropic_smr_type, gyrotropic_smr_fixed_en_width = gyrotropic_smr_fixed_en_width * common.CONSTANTS['kB[eV/K]'])

class Win:
    '''
    Class of seedname.win.

    Args:
        system (System):
        jobcontrol (JobControl):
        disentangle (Disentangle):
        wannierise (Wannierise):
        plot (Plot):

    Attributes:
        structure (Structure):
    '''
    def __init__(self,
                 system: System,
                 jobcontrol: JobControl,
                 disentangle: Disentangle,
                 wannierise: Wannierise,
                 plot: Plot,
                 postw90: Postw90,
                 berry: Berry,
                 gyrotropic: Gyrotropic):
        self.system = system
        self.jobcontrol = jobcontrol
        self.disentangle = disentangle
        self.wannierise = wannierise
        self.plot = plot
        self.postw90 = postw90
        self.berry = berry
        self.gyrotropic = gyrotropic
    def __str__(self):
        lines = [str(self.berry),
                 str(self.gyrotropic),
                 str(self.disentangle),
                 str(self.plot),
                 str(self.wannierise),
                 str(self.system),
                 str(self.jobcontrol),
                 str(self.postw90)]
        return '\n'.join(lines)
    @property
    def structure(self)-> Structure:
        '''
        Structure:
        '''
        return self.system.structure
    @staticmethod
    def __get_block(f: TextIOWrapper)-> list[str]:
        '''
        Method to read block between begin ... and end ....

        Args:
            f (TextIOWrapper):

        Return:
            list[str]: Block between begin ... and end ....
        '''
        block = []
        while True:
            line = f.readline()
            # Stop when end is found.
            if line.startswith('end'):
                break
            # Characters after # and ! are comments.
            line = line.split('#')[0].split('!')[0].strip()
            # Skip blank line.
            if not line:
                continue
            block.append(line)
        return block
    @classmethod
    def from_file(cls,
                  wanndir: Path,
                  seedname: str):
        '''
        Classmethod from seedname.win.

        Args:
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str, optional): Seedname.
        '''
        filename = wanndir / f'{seedname}.win'
        with open(file = filename) as f:
            system = System()
            jobcontrol = JobControl()
            disentangle = Disentangle()
            wannierise = Wannierise()
            plot = Plot()
            postw90 = Postw90()
            berry = Berry()
            gyrotropic = Gyrotropic()
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
                # get_block mode begins (ends) when begin (end) is found.
                if line.startswith('begin'):
                    key = line.split()[1].strip()
                    value = __class__.__get_block(f = f)
                # Parameters are separated with : and =.
                else:
                    line = line.replace(':', '=').split('=')
                    key = line[0].strip()
                    value = line[1].strip()
                if key in System.PARAMS:
                    system.add_param(key = key, value = value)
                elif key in JobControl.PARAMS:
                    jobcontrol.add_param(key = key, value = value)
                elif key in Disentangle.PARAMS:
                    disentangle.add_param(key = key, value = value)
                elif key in Wannierise.PARAMS:
                    wannierise.add_param(key = key, value = value)
                elif key in Plot.PARAMS:
                    plot.add_param(key = key, value = value)
                elif key in Postw90.PARAMS:
                    postw90.add_param(key = key, value = value)
                elif key in Berry.PARAMS:
                    berry.add_param(key = key, value = value)
                elif key in Gyrotropic.PARAMS:
                    gyrotropic.add_param(key = key, value = value)
        print(f'## Reading {filename} finished. ##')
        return cls(system = system, jobcontrol = jobcontrol, disentangle = disentangle, wannierise = wannierise, plot = plot,
                   postw90 = postw90, berry = berry, gyrotropic = gyrotropic)
    def append(self,
               eig: Eig,
               efermi: float,
               bands_num_points: int,
               num_iter: int,
               dis_win_max: float = None,
               dis_froz_max: float = None,
               guiding_centres: bool = True):
        '''
        Method to append disentangle, wannierise, and plot parameters from Eig object.

        Args:
            eig (Eig):
            efermi (float): Pristine Fermi energy [eV].
            bands_num_points (int): Number of k-points per line.
            num_iter (int): Number of wannierise iterations.
            dis_win_max (float, optional): Upper bound of outer window [eV] measured from pristine Fermi energy. Defaults to None.
            dis_froz_max (float, optional): Upper bound of inner window [eV] measured from pristine Fermi energy. Defaults to None.
            guiding_centres (bool, optional): Whether use guiding centers or not. Defaults to True.
        '''
        self.disentangle = Disentangle.from_eig(eig = eig, num_wann = self.system.params['num_wann'],
                                                efermi = efermi, dis_win_max = dis_win_max, dis_froz_max = dis_froz_max)
        self.plot = Plot.from_structure(structure = self.system.structure, bands_num_points = bands_num_points)
        self.wannierise = Wannierise(guiding_centres = guiding_centres, num_iter = num_iter)
    def append_spin_hall(self,
                         berry: bool,
                         gyrotropic: bool,
                         efermi: float,
                         fermi_energy_range: list[float],
                         kmesh: list[int],
                         component: common.Component,
                         fermi_energy_step: float = 0.05):
        '''
        Method to append berry and gyrotropic parameters to compute spin Hall conductivity and spin accumulation coefficient.

        Args:
            berry (bool): Use berry module or not.
            gyrotropic (bool): Use gyrotropic module or not.
            efermi (float): Pristine Fermi energy [eV].
            fermi_energy_range (list[float]): Range of Fermi energy [eV] measured from pristine Fermi energy.
            kmesh (list[int]): Number of k-points for berry and gyrotropic calculations. Length is 3.
            component (common.Component): Of spin Hall conductivity.
            fermi_energy_step (float, optional): Increment of Fermi energy [eV]. Defaults to 0.05.
        '''
        # Translate component to Component object if str.
        if isinstance(component, str):
            component = common.Component(component = component)
        self.plot.add_param('fermi_energy_min', fermi_energy_range[0] + efermi)
        self.plot.add_param('fermi_energy_max', fermi_energy_range[1] + efermi)
        self.plot.add_param('fermi_energy_step', fermi_energy_step)
        self.postw90 = Postw90()
        self.berry = Berry.spin_hall(berry = berry, berry_kmesh = kmesh, component = component)
        self.gyrotropic = Gyrotropic.spin_hall(gyrotropic = gyrotropic, gyrotropic_kmesh = kmesh)
    def write_file(self,
                   wanndir: Path,
                   seedname: str):
        '''
        Method to write to seedname.win.

        Args:
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str, optional): Seedname.
        '''
        filename = wanndir / f'{seedname}.win'
        with open(file = filename, mode = 'w') as f:
            f.write(str(self))
        print(f'## Saving {filename} finished. ##')

def get_jobscript(seedname: str,
                  queue_name: str,
                  num_procs: int,
                  pe_name: str = None)-> common.JobScript:
    '''
    Method to get common.JobScript object for wannier90.x.

    Args:
        seedname (str, optional): Seedname.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

    Returns:
        common.JobScript:
    '''
    job_name = 'wannier90'
    exe_name = CLUSTER['directories'][_PACKAGE] / f'{job_name}.x'
    return common.JobScript(job_name = job_name, stdout = 'stdout', queue_name = queue_name,
                            num_procs = num_procs, exe_name = exe_name, postfix = seedname, pe_name = pe_name)

class BandDat:
    '''
    Class of seedname_band.dat.

    Args:
        distances (list[float]): Distances [AA^{-1}]. Length is num_kpoints.
        eigvals (np.ndarray[float]): Eigenvalues [eV]. Shape is (num_wann, num_kpoints).

    Attributes:
        num_kpoints (int): Number of k-points.
        num_wann (int): Number of Wannier orbitals.
    '''
    def __init__(self,
                 distances,
                 eigvals: np.ndarray[float]):
        self.distances = distances
        self.eigvals = eigvals
    @property
    def num_kpoints(self)-> int:
        '''
        Number of k-points.
        '''
        return len(self.distances)
    @property
    def num_wann(self)-> int:
        '''
        Number of Wannier orbitals.
        '''
        return len(self.eigvals)
    @classmethod
    def from_file(cls,
                  wanndir: Path,
                  seedname: str):
        '''
        Classmethod from seedname_band.dat.

        Args:
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str): Seedname.
        '''
        filename = wanndir / f'{seedname}_band.dat'
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
                    # Format: distance, eigval.
                    distances.append(float(line[0]))
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
    def get_bandstructure(self,
                          efermi: float)-> common.BandStructure:
        '''
        Method to get common.BandStructure object.

        Args:
            efermi (float): Pristine Fermi energy [eV].
        '''
        # Subtract pristine Fermi energy.
        return common.BandStructure(distances = self.distances, eigvals = self.eigvals - efermi)

class HrDat:
    '''
    Class of seedname_hr.dat

    Args:
        degs (list[int]): Degeneracies of Wigner-Seitz grid-points.
        rpts (np.ndarray[int]): Wigner-Seitz grid-points. Shape is (nrpts, 3).
        hrs (np.ndarray[complex]): Hamiltonian matrix elements [eV]. Shape is (nrpts, num_wann, num_wann).

    Attributes:
        nrpts (int): Number of Wigner-Seitz grid-points.
        num_wann (int): Number of Wannier orbitals.
    '''
    def __init__(self,
                 degs: list[int],
                 rpts: np.ndarray[int],
                 hrs: np.ndarray[float]):
        self.degs = degs
        self.rpts = rpts
        self.hrs = hrs
    @property
    def nrpts(self)-> int:
        '''
        Number of Wigner-Seitz grid-points.
        '''
        return self.hrs.shape[0]
    @property
    def num_wann(self)-> int:
        '''
        Number of Wannier orbitals.
        '''
        return self.hrs.shape[1]
    @classmethod
    def from_file(cls,
                  wanndir: Path,
                  seedname: str):
        '''
        Classmethod from seedname_hr.dat.

        Args:
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str): Seedname.
        '''
        filename = wanndir / f'{seedname}_hr.dat'
        with open(file = filename) as f:
            # Skip comment line.
            line = f.readline()
            line = f.readline()
            line = line.strip()
            num_wann = int(line)
            line = f.readline()
            line = line.strip()
            nrpts = int(line)
            degs = []
            for _ in range(0, nrpts, 15):
                line = f.readline()
                line = line.strip()
                line = line.split()
                degs.extend(int(deg)
                            for deg in line)
            rpts = np.empty(shape = (nrpts, 3), dtype = int)
            hrs = np.empty(shape = (nrpts, num_wann, num_wann), dtype = complex)
            for r in range(nrpts):
                for _ in range(num_wann):
                    for _ in range(num_wann):
                        line = f.readline()
                        line = line.strip()
                        line = line.split()
                        hrs[r, int(line[3]) - 1, int(line[4]) - 1] = float(line[5]) + 1j * float(line[6])
                rpts[r] = [int(rpt)
                           for rpt in line[0:3]]
        print(f'## Reading {filename} finished. ##')
        return cls(degs = degs, rpts = rpts, hrs = hrs)
    def get_hamiltonian(self,
                        kvec: np.ndarray[float])-> np.ndarray[complex]:
        '''
        Method to get Wannier-gauge Hamiltonian matrix elements.

        Args:
            kvec (np.ndarray[float]): Fractional k-point. Shape is (3,).

        Returns:
            np.ndarray[complex]: Wannier-gauge Hamiltonian matrix elements [eV].
        '''
        phases = np.exp(1j * 2 * np.pi * np.dot(a = self.rpts, b = kvec)) / self.degs
        return np.tensordot(a = self.hrs, b = phases, axes = ((0,), (0,)))

class CarrierNumber:
    '''
    Class to compute carrier number and Fermi energy.

    Args:
        eigvals (np.ndarray[float]): Eigenvalues [eV]. Shape is (num_bands, num_kpoints).

    Attributes:
        num_wann (int): Number of Wannier orbitals.
        num_kpoints (int): Number of k-points.
        emax (float): Maximum eigenvalue [eV].
        emin (float): Minimum eigenvalue [eV].
    '''
    # Always converges after about 20 iterations because 2^{-20} = 0.95 * 10^{-6}.
    EPS = 1e-6
    def __init__(self,
                 eigvals: np.ndarray[float]):
        self.eigvals = eigvals
    @property
    def num_wann(self)-> int:
        '''
        Number of Wannier orbitals.
        '''
        return self.eigvals.shape[0]
    @property
    def num_kpoints(self)-> int:
        '''
        Number of k-points.
        '''
        return self.eigvals.shape[1]
    @property
    def emax(self)-> float:
        '''
        Maximum eigenvalue [eV].
        '''
        return np.amax(a = self.eigvals)
    @property
    def emin(self)-> float:
        '''
        Minimum eigenvalue [eV].
        '''
        return np.amin(a = self.eigvals)
    @classmethod
    def from_wannier90(cls,
                       wanndir: Path,
                       seedname: str,
                       kmesh: list[int],
                       shifts: list[int]):
        hrdat = HrDat.from_file(wanndir = wanndir, seedname = seedname)
        kpoints = Vectors.monkhorst_pack(kmesh = kmesh, shifts = shifts)
        eigvals = np.empty(shape = (hrdat.num_wann, kpoints.num_vectors), dtype = float)
        for (ik, kvec) in enumerate(kpoints):
            hk = hrdat.get_hamiltonian(kvec = kvec)
            eigvals[:, ik] = np.linalg.eigvalsh(a = hk)
        return cls(eigvals = eigvals)
    def get_number(self,
                   efermi: float)-> float:
        '''
        Method to compute carrier number from Fermi energy.

        Args:
            efermi (float): Fermi energy [eV].

        Returns:
            float: Carrier number.
        '''
        return self.eigvals[self.eigvals < efermi].size / self.num_kpoints
    def get_efermi(self,
                   ne: float)-> float:
        '''
        Method to compute Fermi energy from carrier number.

        Args:
            ne (float): Carrier number.

        Returns:
            float: Fermi energy [eV].
        '''
        erange = [self.emin, self.emax]
        # Bisection method.
        while True:
            ef = 0.5 * sum(erange)
            if erange[1] - erange[0] < __class__.EPS:
                return ef
            n = self.get_number(efermi = ef)
            if n < ne:
                erange[0] = ef
            else:
                erange[1] = ef
