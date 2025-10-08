'''
Module to deal with Postw90 output files.
'''

import numpy as np
from pathlib import Path
from abinitio import CLUSTER, common

_PACKAGE = 'wannier90'

class FermiscanDat:
    '''
    Class of seedname-key-fermiscan.dat.

    Args:
        key (str): 'ahc', 'morb', 'shc'.
        fermi_energies (list[float]): Fermi energies [eV]. Length is nfermis.
        values (np.ndarray[float]):
            'ahc': Anomalous Hall conductivity [S/cm]. Shape is (3, nfermis).\\
            'morb': Orbital magnetization [mu_{B}]. Shape is (3, nfermis).\\
            'shc': Spin Hall conductivity [(hbar/e) S/cm]. Shape is (nfermis,).

    Attributes:
        nfermis (int): Number of Fermi energies.
    '''
    def __init__(self,
                 key: str,
                 fermi_energies: list[float],
                 values: np.ndarray[float]):
        self.key = key
        self.fermi_energies = fermi_energies
        self.values = values
    @property
    def nfermi(self)-> int:
        '''
        Number of Fermi energies.
        '''
        return len(self.fermi_energies)
    @classmethod
    def from_file(cls,
                  key: str,
                  wanndir: Path,
                  seedname: str):
        '''
        Classmethod from seedname-key-fermiscan.dat.

        Args:
            key (str): 'ahc', 'morb', 'shc'.
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str): Seedname.
        '''
        filename = wanndir / f'{seedname}-{key}-fermiscan.dat'
        with open(file = filename) as f:
            # Skip comment line for shc.
            if key == 'shc':
                line = f.readline()
            fermi_energies = []
            values = []
            while True:
                line = f.readline()
                # Stop at end of file.
                if not line:
                    break
                line = line.strip().split()
                if key in ['ahc', 'morb']:
                    # Format: fermi_energy, value(3) for ahc, morb.
                    fermi_energies.append(float(line[0]))
                    values.append(line[1:4])
                elif key == 'shc':
                    # Format: index, fermi_energy, value for shc.
                    fermi_energies.append(float(line[1]))
                    values.append(line[2])
        print(f'## Reading {filename} finished. ##')
        values = np.array(object = values, dtype = float)
        # Transpose from (nefermis, 3) to (3, nfermis) for ahc, morb.
        if key in ['ahc', 'morb']:
            values = values.transpose()
        return cls(key = key, fermi_energies = fermi_energies, values = values)
    def get_physicalquantity(self,
                             efermi: float,
                             label: str = None,
                             *component: int):
        '''
        Method to get common.PhysicalQuantity object.

        Args:
            efermi (float): Pristine Fermi energy [eV].
            label (str, optional): Label.
            *component (int): Component 0, 1, 2 for ahc, morb.

        Returns:
            common.PhysicalQuantity:
        '''
        # Subtract pristine Fermi energy.
        fermi_energies = [fermi_energy - efermi for fermi_energy in self.fermi_energies]
        if self.key in ['ahc', 'morb']:
            if not component:
                raise ValueError('Specify component among 0, 1, 2.')
            values = self.values[component[0]]
        elif self.key == 'shc':
            values = self.values
        return common.PhysicalQuantity(key = self.key, fermi_energies = fermi_energies, values = values, label = label)

class GyrotropicDat:
    '''
    Class of seedname-gyrotropic-key.dat.

    Args:
        key (str): 'acc_spin', 'acc_spin_x', 'acc_spin_y', 'acc_spin_z',
            'C', 'D', 'DOS', 'K_orb', 'K_spin', 'NOA_orb', 'NOA_spin', 'tildeD'.
        fermi_energies (list[float]): Fermi energies [eV]. Length is nfermis.
        frequencies (list[float]): Frequencies [eV]. Length is nfreqs.
        values (np.ndarray[float]):
            'acc_spin': Spin accumulation coefficient [(hbar/e) S/cm]. Shape is (3, 3, 3, nfermis).\\
            'acc_spin_{x, y, z}': Spin accumulation coefficient [(hbar/e) S/cm] of spin // {x, y, z}. Shape is (3, 3, nfermis).\\
            'C': Electric conductivity [A/cm]. Shape is (3, 3, nfermis).\\
            'D': Berry curvature dipole. Shape is (3, 3, nfermis).\\
            'DOS': Density of states [eV^{-1} AA^{-3}]. Shape is (nfermis,).\\
            'K_{orb, spin}': {Orbital, spin} Edelstein coefficient [A]. Shape is (3, 3, nfermis).\\
            'NOA_{orb, spin}': {Orbital, spin} natural optical activity [AA]. Shape is (3, 3, nfreqs, nfermis).\\
            'tildeD': ac Berry curvature dipole. Shape is (3, 3, nfreqs, nfermis).
    
    Attributes:
        nfermis (int): Number of Fermi energies.
        nfreqs (int): Number of frequencies.
    '''
    # Rank-2 tensor or not, symmetrized or not, ac or dc.
    __INFO = {
        'acc_spin': [False, False, False],
        'acc_spin_x': [True, False, False],
        'acc_spin_y': [True, False, False],
        'acc_spin_z': [True, False, False],
        'C': [True, True, False],
        'D': [True, True, False],
        'DOS': [False, False, False],
        'K_orb': [True, True, False],
        'K_spin': [True, True, False],
        'NOA_orb': [True, False, True],
        'NOA_spin': [True, False, True],
        'tildeD': [True, True, True]
    }
    def __init__(self,
                 key: str,
                 fermi_energies: list[float],
                 frequencies: list[float],
                 values: np.ndarray[float]):
        self.key = key
        self.fermi_energies = fermi_energies
        self.frequencies = frequencies
        self.values = values
    @property
    def nfermis(self)-> int:
        '''
        Number of Fermi energies.
        '''
        return len(self.fermi_energies)
    @property
    def nfreqs(self)-> int:
        '''
        Number of frequencies.
        '''
        return len(self.frequencies)
    @classmethod
    def from_file(cls,
                  key: str,
                  wanndir: Path,
                  seedname: str):
        '''
        Classmethod from seedname-gyrotropic-key.dat.

        Args:
            key (str): 'acc_spin_x', 'acc_spin_y', 'acc_spin_z',
                'C', 'D', 'DOS', 'K_orb', 'K_spin', 'NOA_orb', 'NOA_spin', 'tildeD'.
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str): Seedname.
        '''
        filename = wanndir / f'{seedname}-gyrotropic-{key}.dat'
        with open(file = filename) as f:
            # Skip 2 comment lines.
            for _ in range(2):
                line = f.readline()
            frequencies = []
            values = []
            # Loop over frequencies.
            while True:
                line = f.readline()
                line = line.strip()
                # Stop at blank line.
                if not line:
                    break
                # Skip 1 more comment line if symmetrized.
                if __class__.__INFO[key][1]:
                    line = f.readline()
                fermi_energies = []
                # Loop over fermi_energies.
                while True:
                    line = f.readline()
                    line = line.strip()
                    # Break at 2 blank lines.
                    if not line:
                        line = f.readline()
                        break
                    line = line.split()
                    fermi_energies.append(float(line[0]))
                    # For rank-2 tensors.
                    if __class__.__INFO[key][0]:
                        # Format: fermi_energy, frequency, value(9).
                        frequencies.append(float(line[1]))
                        values.append(line[2:11])
                    # For rank-0 tensors, namely, DOS.
                    else:
                        # Format: fermi_energy, value.
                        values.append(line[1])
        print(f'## Reading {filename} finished. ##')
        values = np.array(object = values, dtype = float)
        # For rank-2 tensors.
        if __class__.__INFO[key][0]:
            tensors = np.empty(shape = (3, 3, len(values)), dtype = float)
            # Transpose from (*, 9) to (9, *).
            values = values.transpose()
            # Sort symmetrized tensors, namely, C, D, K_orb, K_spin, tildeD.
            # [xx, yy, zz, (xy + yx) / 2, (xz + zx) / 2, (yz + zy) / 2, (yz - zy) / 2, (zx - xz) / 2, (xy - yx) / 2]
            # -> [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]].
            if __class__.__INFO[key][1]:
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            tensors[i, i] = values[i]
                        elif [i, j] in [[0, 1], [1, 2], [2, 0]]:
                            tensors[i, j] = values[2 + i + j] + values[9 - i - j]
                        # [i, j] in [[1, 0], [2, 1], [0, 2]].
                        else:
                            tensors[i, j] = values[2 + i + j] - values[9 - i - j]
            # Sort unsymmetrized tensors.
            # [xx, yy, zz, xy, xz, yz, zy, zx, yx]
            # -> [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]].
            else:
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            tensors[i, i] = values[i]
                        elif i < j:
                            tensors[i, j] = values[2 + i + j]
                        # i > j.
                        else:
                            tensors[i, j] = values[9 - i - j]
            # Reshape (3, 3, *) to (3, 3, nfreqs, nfermis) for ac quantities, namely, NOA_orb, NOA_spin, tildeD.
            nfermis = len(fermi_energies)
            frequencies = frequencies[::nfermis]
            if __class__.__INFO[key][2]:
                tensors = tensors.reshape(3, 3, -1, nfermis)
        # For rank-0 tensors, namely, DOS.
        else:
            frequencies = [0.0]
            tensors = values
        return cls(key = key, fermi_energies = fermi_energies, frequencies = frequencies, values = tensors)
    @classmethod
    def acc_spin_from_file(cls,
                           wanndir: Path,
                           seedname: str):
        '''
        Classmethod from seedname-gyrotropic-acc_spin_{x, y, z}.dat.

        Args:
            wanndir (Path): Working directory for Wannier90 calculations.
            seedname (str): Seedname.
        '''
        values = []
        for a in range(3):
            # ASCII code 120-> x, 121-> y, 122-> z.
            key = 'acc_spin_' + chr(120 + a)
            gyrotropicdat = __class__.from_file(key = key, wanndir = wanndir, seedname = seedname)
            fermi_energies = gyrotropicdat.fermi_energies
            frequencies = gyrotropicdat.frequencies
            values.append(gyrotropicdat.values)
        values = np.array(object = values)
        return cls(key = 'acc_spin', fermi_energies = fermi_energies, frequencies = frequencies, values = values)
    def get_physicalquantity(self,
                             efermi: float,
                             label: str = None,
                             *component: int):
        '''
        Method to get common.PhysicalQuantity object.

        Args:
            efermi (float): Pristine Fermi energy [eV].
            label (str, optional): Label.
            *component (tuple[int]): Component.

        Returns:
            common.PhysicalQuantity:
        '''
        # Subtract pristine Fermi energy.
        fermi_energies = [fermi_energy - efermi for fermi_energy in self.fermi_energies]
        if not component:
            raise ValueError('Specify component.')
        values = self.values[component]
        return common.PhysicalQuantity(key = self.key, fermi_energies = fermi_energies, values = values, label = label)
    def get_physicalquantity_w_errors(self,
                                      efermi: float,
                                      weights: dict[common.Component, float],
                                      label: str = None):
        '''
        Method to get common.PhysicalQuantity object with errors attribute.

        Args:
            efermi (float): Pristine Fermi energy [eV].
            weights (dict[common.Component, float]): Weights of symmetry-allowed components.
            label (str, optional): Label. Defaults to None.

        Returns:
            common.PhysicalQuantity:
        '''
        # Subtract pristine Fermi energy.
        fermi_energies = [fermi_energy - efermi
                          for fermi_energy in self.fermi_energies]
        weighted_values = [self.values[component] * weight
                           for (component, weight) in weights.items()]
        averages = np.average(a = weighted_values, axis = 0)
        errors = np.amax(a = np.abs(weighted_values - averages), axis = 0)
        return common.PhysicalQuantity(key = self.key, fermi_energies = fermi_energies, values = averages, label = label, errors = errors)

def get_jobscript(seedname: str,
                  queue_name: str,
                  num_procs: int,
                  pe_name: str = None)-> common.JobScript:
    '''
    Method to get common.JobScript object for postw90.x.

    Args:
        seedname (str, optional): Seedname.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.

    Returns:
        common.JobScript:
    '''
    job_name = 'postw90'
    exe_name = CLUSTER['directories'][_PACKAGE] / 'postw90.x'
    return common.JobScript(job_name = job_name, stdout = 'stdout', queue_name = queue_name,
                            num_procs = num_procs, exe_name = exe_name, postfix = seedname, pe_name = pe_name)
