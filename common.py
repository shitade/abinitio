'''
Module of common utilities.
Save colors.jpg for color setting.

Usage:
    >> python3 <Path/to/common.py>
'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure
from abinitio import HOME, CLUSTER

# Universal constants.
CONSTANTS = {
    'Ry[eV]': 13.60569312,
    'kB[eV/K]': 8.617333262e-5,
    'Bohr[AA]': 0.5291772105,
    'inch[cm]': 0.3937007874,
    'gold': (1 + np.sqrt(5)) / 2
}

def __rgb(r: int,
          g: int,
          b: int)-> tuple[float]:
    '''
    Method to get RGB for matplotlib.

    Args:
        r (int): 0 <= r <= 255.
        g (int): 0 <= g <= 255.
        b (int): 0 <= b <= 255.

    Returns:
        tuple[float]: RGB for matplotlib.
    '''
    if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
        return (r / 255, g / 255, b / 255)
    else:
        raise ValueError('0 <= R, G, B <= 255.')

GRID = {
    'color': __rgb(128, 128, 128), # Gray.
    'linestyle': (0, ()), # Solid or '-'.
    'linewidth': 0.5
}
# Wong, Nat. Meth. 8, 441 (2011). https://doi.org/10.1038/s41592-023-01974-0
LINESTYLES = [
    {
        'color': __rgb(0, 0, 0), # Black.
        'linestyle': (0, ()) # Solid or '-'.
    },
    {
        'color': __rgb(230, 159, 0), # Orange.
        'linestyle': (0, (1, 1)) # Dotted or ':'.
    },
    {
        'color': __rgb(86, 180, 233), # Sky blue.
        'linestyle': (0, (3, 1)) # Dashed.
    },
    {
        'color': __rgb(0, 158, 115), # Bluish green.
        'linestyle': (0, (3, 1, 1, 1)) # Dash-dotted.
    },
    {
        'color': __rgb(240, 228, 66), # Yellow.
        'linestyle': (0, (3, 1, 1, 1, 1, 1)) # Dash-dot-dotted.
    },
    {
        'color': __rgb(0, 114, 178), # Blue.
        'linestyle': (0, (5, 1)) # Long-dashed.
    },
    {
        'color': __rgb(213, 94, 0), # Vermillion.
        'linestyle': (0, (5, 1, 1, 1)) # Long-dash-dotted.
    },
    {
        'color': __rgb(204, 121, 167), # Reddish purple.
        'linestyle': (0, (5, 1, 1, 1, 1, 1)) # Long-dash-dot-dotted.
    }
]
POINTSTYLES = [
    {
        'color': __rgb(0, 0, 0), # Black.
        'marker': 'o', 'markerfacecolor': 'none', # Open-circle.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color': __rgb(230, 159, 0), # Orange.
        'marker': 's', 'markerfacecolor': 'none', # Open-square.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color': __rgb(86, 180, 233), # Sky blue.
        'marker': '^', 'markerfacecolor': 'none', # Open-triangle.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color': '#009E73', # Bluish green: 0, 158, 115
        'marker': 'o', # Filled-circle.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color': __rgb(240, 228, 66), # Yellow.
        'marker': 's', # Filled-square.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color':  __rgb(0, 114, 178), # Blue.
        'marker': '^', # Filled-triangle.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color': __rgb(213, 94, 0), # Vermillion.
        'marker': 'x', # x.
        'linewidth': 0, 'markersize': 2
    },
    {
        'color': __rgb(204, 121, 167), # Reddish purple.
        'marker': '+', # +.
        'linewidth': 0, 'markersize': 2
    }
]

def get_structure_from_materials_project(id: int)-> Structure:
    '''
    Method to get Structure object from Materials Project.

    Args:
        id (int): Material ID in Materials Project.

    Returns:
        Structure: of conventional unit cell.
    '''
    from mp_api.client import MPRester
    with MPRester() as mpr:
        # Get Structure object of conventional unit cell using MPRester.get_structure_by_material_id with conventional_unit_cell = True.
        # Then, write cif file.
        # Before caclulations, get Structure object of primitive unit cell using Structure.from_file with primitive = True.
        # This is better than getting Structure object of primitive unit cell
        # using MPRester.get_structure_by_material_id without conventional_unit_cell = True.
        structure = mpr.get_structure_by_material_id(material_id = f'mp-{id}', conventional_unit_cell = True)
    return structure

def get_structure_from_cif(dir: Path,
                           mater: str,
                           magmom: list[float] = None)-> Structure:
    '''
    Method to get Structure object from cif file.

    Args:
        dir (Path): Working directory.
        mater (str): Prefix of cif file.
        magmom (list[float], optional): Magnetic moments [mu_{B}] of sites. Defaults to None.

    Returns:
        Structure: of primitive unit cell.
    '''
    filename = dir / f'{mater}.cif'
    structure = Structure.from_file(filename = filename, primitive = True)
    print(f'## Reading {filename} finished. ##')
    if magmom is None:
        magmom = [0] * structure.num_sites
    structure.add_site_property(property_name = 'magmom', values = magmom)
    return structure

class BandStructure:
    '''
    Class of band structure.

    Args:
        distances (list[float]): Distances [AA^{-1}]. Length is num_kpoints.
        eigvals (np.ndarray[float]): Eigenvalues [eV]. Shape is (num_bands, num_kpoints).

    Attributes:
        num_bands (int): Number of bands.
        num_kpoints (int): Number of k-points.
    '''
    def __init__(self,
                 distances: list[float],
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

class BandPlotter:
    '''
    Class to plot band structure.

    Args:
        xticklabels (list[str]): x-ticklabels.
        xticks (list[float]): x-ticks [AA^{-1}].
        bandstructures (list[BandStructure], optional): Defaults to None, leading to [].
    '''
    __WIDTH = 14 # [cm]
    __FIGSIZE = np.array(object = [1, 1 / CONSTANTS['gold']], dtype = float) * __WIDTH * CONSTANTS['inch[cm]']
    __DPI = 600
    def __init__(self,
                 xticklabels: list[str],
                 xticks: list[float],
                 bandstructures: list[BandStructure] = None):
        self.xticklabels = xticklabels
        self.xticks = xticks
        self.bandstructures = bandstructures if bandstructures is not None else []
    @classmethod
    def from_structure(cls,
                       structure: Structure):
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        xticklabels = []
        xticks = []
        metric = structure.lattice.reciprocal_lattice.metric_tensor
        kpath = HighSymmKpath(structure = structure, has_magmoms = True, magmom_axis = [0, 0, 1]).kpath
        for line in kpath['path']:
            for i in range(len(line)):
                label = f'${line[i]}$'
                if i == 0:
                    # Origin.
                    if not xticklabels:
                        xticklabels.append(label)
                        xticks.append(0.0)
                    # Discontinuous path.
                    else:
                        xticklabels[-1] = xticklabels[-1] + '|' + label
                else:
                    xticklabels.append(label)
                    delta_frac = kpath['kpoints'][line[i]] - kpath['kpoints'][line[i - 1]]
                    distance = np.sqrt(delta_frac @ metric @ delta_frac)
                    xticks.append(xticks[-1] + distance)
        return cls(xticklabels = xticklabels, xticks = xticks)
    def add_bandstructure(self,
                          bandstructure: BandStructure):
        '''
        Append BandStructure object to bandstructures attribute.

        Args:
            bandstructure (BandStructure):
        '''
        self.bandstructures.append(bandstructure)
    def set_axes(self,
                 ax: Axes,
                 ylim: list[float] = None):
        '''
        Method to set Axes object.

        Args:
            ax (Axes):
            ylim (list[float], optional): Ylim [eV] of each Axes object. Length is 2. Defaults to None.
        '''
        for (bandstructure, linestyle) in zip(self.bandstructures, LINESTYLES):
            for eigval in bandstructure.eigvals:
                ax.plot(bandstructure.distances, eigval, **linestyle)
        ax.axhline(**GRID)
        ax.grid(axis = 'x', **GRID)
        ax.set_xlim(left = self.xticks[0], right = self.xticks[-1])
        ax.set_xticks(ticks = self.xticks, labels = self.xticklabels)
        ax.set_ylabel(ylabel = 'Energy [eV]')
        ax.set_ylim(ylim)
    def save_plot(self,
                  filename: Path,
                  ylim: list[float] = None):
        '''
        Method to save image of band structure plot.

        Args:
            filename (Path): Filename of band structure plot.
            ylim (list[float], optional): Ylim [eV] of each Axes object. Length is 2. Defaults to None.
        '''
        plt.rcParams['font.size'] = __class__.__WIDTH
        (fig, ax) = plt.subplots(figsize = __class__.__FIGSIZE, dpi = __class__.__DPI)
        self.set_axes(ax = ax, ylim = ylim)
        fig.tight_layout()
        fig.savefig(fname = filename)
        print(f'## Saving {filename} finished. ##')

class Component(tuple[int]):
    '''
    Class of component of band and physical properties.

    Args:
        *component (int): Component.

    Attributes:
        to_xyz (str): For instance, xyz for (0, 1, 2).
        to_012 (str): For instance, 012 for (0, 1, 2).
        to_123 (str): For instance, 123 for (0, 1, 2).
    '''
    def __new__ (cls, *component):
        return super().__new__(cls, component)
    @property
    def to_xyz(self)-> str:
        '''
        For instance, xyz for (0, 1, 2).
        '''
        # ASCII code 120 = x, 121 = y, 122 = z.
        return ''.join(chr(120 + arg)
                       for arg in self)
    @property
    def to_012(self)-> str:
        '''
        For instance, 012 for (0, 1, 2).
        '''
        return ''.join(str(arg)
                       for arg in self)
    @property
    def to_123(self)-> str:
        '''
        For instance, 123 for (0, 1, 2).
        '''
        return ''.join(str(arg + 1)
                       for arg in self)

class PhysicalQuantity:
    '''
    Class of physical quantity.

    Args:
        key (str): 'ahc', 'morb', 'shc', 'acc_spin', 'C', 'D', 'DOS', 'K_orb', 'K_spin', 'NOA_orb', 'NOA_spin', 'tildeD'.
        fermi_energies (list[float]): Fermi energies [eV]. Length is nfermis.
        values (list[float]): Physical quantity. Length is nfermis.
        label (str, optional): Label. Defaults to None.
        errors (list[float], optional): Errors of physical quantity with legnth nfermis. Defaults to None.

    Attributes:
        name (str): Name.
        symbol (str): Mathematical symbol.
        unit (str): Unit.
        ylabel (str): Default ylabel when plotted.
        nfermis (int): Number of Fermi energies.
    '''
    # Name, symbol, unit.
    INFO = {
        'ahc': ['Anomalous Hall conductivity', r'$\sigma$', 'S/cm'],
        'morb': ['Orbital magnetization', r'$M_{o}$', r'$\mu_{\mathrm{B}}$'],
        'shc': ['Spin Hall conductivity', r'$\sigma_{s}$', r'($\hbar/e$) S/cm'],
        'acc_spin': ['Spin accumulation coefficient', r'$\gamma_{s}$', r'($\hbar/e$) S/cm'],
        'C': ['Electric conductivity', '$C$', 'A/cm'],
        'D': ['Berry curvature dipole', '$D$', None],
        'DOS': ['Density of states', 'DOS', r'$\mathrm{eV}^{-1} \mathrm{\AA}^{-3}$'],
        'K_orb': ['Orbital Edelstein coefficient', r'$\alpha_{o}$', 'A'],
        'K_spin': ['Spin Edelstein coefficient', r'$\alpha_{s}$', 'A'],
        'NOA_orb': ['Orbital natural optical activity', r'$\gamma_{o}$', r'$\mathrm{\AA}$'],
        'NOA_spin': ['Spin natural optical activity', r'$\gamma_{s}$', r'$\mathrm{\AA}$'],
        'tildeD': ['Berry curvature dipole', '$D$', None]
    }
    def __init__(self,
                 key: str,
                 fermi_energies: list[float],
                 values: list[float],
                 label: str = None,
                 errors: list[float] = None):
        self.key = key
        self.fermi_energies = fermi_energies
        self.values = values
        self.label = label
        self.errors = errors
    def __str__(self):
        lines = [f'## {self.name} ##',
                 f'## Fermi energy [eV], {self.symbol} [{self.unit}] ##']
        if self.errors is None:
            for (fermi_energy, value) in zip(self.fermi_energies, self.values):
                lines.append(f'{fermi_energy} {value}')
        else:
            for (fermi_energy, value, error) in zip(self.fermi_energies, self.values, self.errors):
                lines.append(f'{fermi_energy} {value} {error}')
        return '\n'.join(lines)
    @property
    def name(self)-> str:
        '''
        Name.
        '''
        return __class__.INFO[self.key][0]
    @property
    def symbol(self)-> str:
        '''
        Mathematical symbol.
        '''
        return __class__.INFO[self.key][1]
    @property
    def unit(self)-> str:
        '''
        Unit.
        '''
        return __class__.INFO[self.key][2]
    @property
    def ylabel(self)-> str:
        '''
        Default ylabel when plotted.
        '''
        # Format: symbol.
        ylabel = self.symbol
        if self.unit is not None:
            # Format: symbol [unit].
            ylabel = ylabel + f' [{self.unit}]'
        return ylabel
    @property
    def nfermis(self)-> int:
        '''
        Number of Fermi energies.
        '''
        return len(self.fermi_energies)
    @classmethod
    def average(cls,
                physicalquantities: dict[Component, 'PhysicalQuantity'],
                weights: dict[Component, float],
                label: str = None):
        '''
        Classmethod to average of PhysicalQuantity objects with some weights.

        Args:
            physicalquantities (dict[Component, PhysicalQuantity]): PhysicalQuantity objects of symmetry-allowed components.
            weights (dict[Component, float]): Weights of symmetry-allowed components.
            label (str, optional): Label. Defaults to None.
        '''
        # Only if keys and fermi_energies of all physical quantities are same.
        key = None
        fermi_energies = None
        for physicalquantity in physicalquantities.values():
            if key is None or fermi_energies is None:
                key = physicalquantity.key
                fermi_energies = physicalquantity.fermi_energies
            else:
                if not (physicalquantity.key == key and np.array_equal(physicalquantity.fermi_energies, fermi_energies)):
                    return None
        weighted_values = [physicalquantities[component].values * weight
                           for (component, weight) in weights.items()]
        averages = np.average(a = weighted_values, axis = 0)
        errors = np.amax(a = np.abs(weighted_values - averages), axis = 0)
        return cls(key = key, fermi_energies = fermi_energies, values = averages, label = label, errors = errors)

class PhysicalPlotter(np.ndarray):
    '''
    Class to plot Fermi-energy dependences of physical quantities.

    Args:
        physicalquantities (tuple[PhysicalQuantity]):
    '''
    __WIDTH = 14 # [cm]
    __FIGSIZE = np.array(object = [1, 1 / CONSTANTS['gold']], dtype = float) * __WIDTH * CONSTANTS['inch[cm]']
    __DPI = 600
    def __new__(cls,
                physicalquantities: tuple[PhysicalQuantity]):
        obj = np.asarray(a = physicalquantities).view(cls)
        if obj.ndim == 1:
            obj = obj.reshape(*obj.shape, 1, 1)
        elif obj.ndim == 2:
            obj = obj.reshape(*obj.shape, 1)
        return obj
    def __array_finalize__(self, obj):
        return super().__array_finalize__(obj)
    @classmethod
    def empty(cls,
              shape: tuple[int]):
        '''
        Classmethod to construct empty object.

        Args:
            shape (tuple[int]): Shape.
        '''
        return cls(np.full(shape = shape, fill_value = None))
    def set_axes(self,
                 axs: np.ndarray[Axes],
                 xlims: dict[Component, list[float]] = None,
                 ylabels: dict[Component, str] = None,
                 ylims: dict[Component, list[float]] = None):
        '''
        Method to set np.ndarray[Axes] object.

        Args:
            axs (np.ndarray[Axes]):
            xlims (dict[Component, list[float], optional): Xlim of each Axes object. Length is 2. Defaults to None.
            ylabels (dict[Component, str]): Ylabel of each Axes object. Defaults to None.
            ylims (dict[Component, list[float], optional): Ylim of each Axes object. Length is 2. Defaults to None.
        '''
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                component = Component(row, col)
                for (physicalquantity, linestyle) in zip(self[component], LINESTYLES):
                    if physicalquantity.errors is None:
                        axs[component].plot(physicalquantity.fermi_energies, physicalquantity.values,
                                                     label = physicalquantity.label, **linestyle)
                    else:
                        axs[component].errorbar(x = physicalquantity.fermi_energies, y = physicalquantity.values,
                                                         yerr = physicalquantity.errors, label = physicalquantity.label, **linestyle)
                axs[component].axhline(**GRID)
                axs[component].axvline(**GRID)
                axs[component].set_xlabel(xlabel = r'$E_{\mathrm{F}}$ [eV]')
                # If xlims is specified.
                if xlims is not None and component in xlims.keys():
                    axs[component].set_xlim(xlims[component])
                # If ylabels is specified.
                if ylabels is not None and component in ylabels.keys():
                    axs[component].set_ylabel(ylabel = ylabels[component])
                else:
                    keys = set(physicalquantity.key for physicalquantity in self[component])
                    ylabel = self[*component, 0].ylabel if len(keys) == 1 else None
                    axs[component].set_ylabel(ylabel = ylabel)
                # If ylims is specified.
                if ylims is not None and component in ylims.keys():
                    axs[component].set_ylim(ylims[component])
                # Set legend only for multiple lines.
                axs[component].legend()
    def save_plot(self,
                  filename: Path,
                  xlims: dict[Component, list[float]] = None,
                  ylabels: dict[Component, str] = None,
                  ylims: dict[Component, list[float]] = None):
        '''
        Method to save image.

        Args:
            filename (Path): Filename.
            xlims (dict[Component, list[float], optional): Xlim of each Axes object. Length is 2. Defaults to None.
            ylabels (dict[Component, str]): Ylabel of each Axes object. Defaults to None.
            ylims (dict[Component, list[float]], optional): Ylim of each Axes object. Length is 2. Defaults to None.
        '''
        plt.rcParams['font.size'] = __class__.__WIDTH
        # shape is (nrows, ncols), namely, (vertical, horizontal),
        # while figsize is (width, height), namely, (horizontal, vertical), and hence reverse shape.
        (fig, axs) = plt.subplots(*self.shape[:2], figsize = __class__.__FIGSIZE * self.shape[1::-1], dpi = __class__.__DPI,
                                  layout = 'tight')
        axs = np.array(object = axs).reshape(*self.shape[:2])
        self.set_axes(axs = axs, xlims = xlims, ylabels = ylabels, ylims = ylims)
        fig.savefig(fname = filename)
        print(f'## Saving {filename} finished. ##')

class JobScript:
    '''
    Class of job script.

    Args:
        job_name (str): Job name.
        stdout (str): Stdout.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        exe_name (Path): Name of execution file.
        postfix (str, optional): Postfix such as seedname. Defaults to None, leading to empty string.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

    Attributes:
        filename (str): Filename, job_name.sh.
    '''
    def __init__(self,
                 job_name: str,
                 stdout: str,
                 queue_name: str,
                 num_procs: int,
                 exe_name: Path,
                 postfix: str = None,
                 pe_name: str = None):
        self.job_name = job_name
        self.stdout = stdout
        self.queue_name = queue_name
        self.num_procs = num_procs
        self.exe_name = exe_name
        if postfix is not None:
            self.postfix = postfix
        else:
            self.postfix = ''
        if pe_name is not None:
            self.pe_name = pe_name
        # Default parallel environment if not specified.
        else:
            self.pe_name = CLUSTER['queues'][queue_name][0]
    def __str__(self):
        lines = ['#!/bin/csh',
                 '#$ -cwd',
                 '#$ -V -S /bin/bash',
                 f'#$ -N {self.job_name}',
                 f'#$ -o {self.stdout}',
                 f'#$ -e {self.stdout}',
                 f'#$ -q {self.queue_name}',
                 f'#$ -pe {self.pe_name} {self.num_procs}',
                 '',
                 f'mpirun -np {self.num_procs} {self.exe_name} {self.postfix}',
                 '']
        return '\n'.join(lines)
    @property
    def filename(self)-> str:
        '''
        Filename, job_name.sh.
        '''
        return self.job_name + '.sh'
    def write_file(self,
                   dir: Path):
        '''
        Method to write job script.

        Args:
            dir (Path): Working directory.
        '''
        filename = dir / self.filename
        with open(file = filename, mode = 'w') as f:
            f.write(str(self))
        print(f'## Saving {filename} finished. ##')

def __save_colors(filename: Path):
    n = 10
    (fig, ax) = plt.subplots()
    ax.axhline(**GRID)
    ax.axvline(**GRID)
    for (i, linestyle) in enumerate(LINESTYLES):
        ax.plot(range(i, i + n), **linestyle, label = str(i))
    for (i, pointstyle) in enumerate(POINTSTYLES):
        ax.plot(range(i, i + n), **pointstyle)
    ax.legend()
    fig.savefig(fname = filename)
    print(f'## Saving {filename} finished. ##')

if __name__ == '__main__':
    __save_colors(filename = HOME / 'colors.jpg')