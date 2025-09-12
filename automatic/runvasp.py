'''
Module to run VASP, Wannier90, and postw90.

Usage:
    >> python3 <Path/to/runvasp.py> -y <YAML> -m <MODE>
'''

from pathlib import Path
from abinitio import HOME, common, vasp, wannier90, postw90, symmetry

__PACKAGE = 'vasp'

def __mkdir(vaspdir: Path):
    '''
    Method to make directory.

    Args:
        vaspdir (Path): Working directroy for VASP calculations to be made.
    '''
    import os
    # Make vaspdir if not exist.
    if not os.path.exists(path = vaspdir):
        print(f'## {vaspdir} does not exist. ##')
        os.makedirs(name = vaspdir)
        print(f'## Making {vaspdir} finished. ##')

def __copy_chgcar_poscar_potcar(scfdir: Path,
                                vaspdir: Path):
    '''
    Method to copy CHGCAR, POSCAR, POTCAR.
    '''
    import shutil
    for file in ['CHGCAR', 'POSCAR', 'POTCAR']:
        shutil.copy2(src = scfdir / file, dst = vaspdir)
        print(f'## Copying {vaspdir / file} finished. ##')

def __write_spn(wanndir: Path):
    '''
    Method to write seedname.spn using wannierberri.

    Args:
        wanndir (Path): Working directory for Wannier90 calculations.
    '''
    import os, subprocess
    seedname = 'wannier90'
    filename = wanndir / f'{seedname}.spn'
    # Write wannier90.spn if not exist.
    if not os.path.isfile(path = filename):
        subprocess.run(args = ['python3', '-m', 'wannierberri.utils.vaspspn'], cwd = wanndir)

def __write_uHu(wanndir: Path,
                extension: str):
    '''
    Method to write seedname.{sHu, sIu, uHu} using wannierberri.

    Args:
        wanndir (Path): Working directory for Wannier90 calculations.
        extension (str): sHu, sIu, uHu.
    '''
    import os, shutil, subprocess
    seedname = 'wannier90'
    filename = wanndir / f'{seedname}.{extension}'
    # Write seedname.suffix if not exist.
    if not os.path.isfile(path = filename):
        subprocess.run(args = ['python3', '-m', 'wannierberri.utils.mmn2uHu', seedname, 'targets=' + extension], cwd = wanndir)
        win = wannier90.Win.from_file(wanndir = wanndir, seedname = seedname)
        num_bands = win.system.params['num_bands']
        newdir = wanndir / f'reduced_NB={num_bands}'
        # Move and rename.
        shutil.move(src = newdir / f'{seedname}_nbs={num_bands}.{extension}',
                    dst = wanndir / f'{seedname}.{extension}')
        # Remove directory.
        shutil.rmtree(path = newdir)

def runscf(mater: str,
           soc: bool,
           kmesh: list[int],
           queue_name: str,
           num_procs: int,
           magmom: list[float] = None,
           vdw: str = None,
           pe_name: str = None):
    '''
    Method to run self-consistent calculations.

    Args:
        mater (str): Material name.
        soc (bool): With or without spin-orbit coupling.
        kmesh (list[int]): Number of k-points for Monkhorst-Pack grids. Length is 3.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        magmom (list[float], optional): Magnetic moments [mu_{B}] of sites. Defaults to None.
        vdw (str, optional): Van der Waals functional. Defaults to None.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    scf = 'scf'
    vaspdir = materdir / scf
    # Make vaspdir if not exist.
    __mkdir(vaspdir = vaspdir)
    structure = common.get_structure_from_cif(dir = materdir, mater = mater, magmom = magmom)
    incarposcar = vasp.IncarPoscar.from_structure(structure = structure, calculation = scf, lsorbit = soc, vdw = vdw)
    # Write INCAR.
    incarposcar.write_incar(vaspdir = vaspdir)
    # Write KPOINTS.
    incarposcar.write_kpoints(vaspdir = vaspdir, kmesh = kmesh)
    # Write POSCAR.
    incarposcar.write_poscar(vaspdir = vaspdir)
    # Write POTCAR.
    incarposcar.write_potcar(vaspdir = vaspdir)
    # Copy vdw_kernel.bindat if necessary.
    incarposcar.copy_vdw(vaspdir = vaspdir)
    jobscript = incarposcar.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write vasp.sh
    jobscript.write_file(dir = vaspdir)
    # >> qsub vasp.sh
    subprocess.run(args = ['qsub', 'vasp.sh'], cwd = vaspdir)

def runbands(mater: str,
             kmesh: int,
             nbands: int,
             queue_name: str,
             num_procs: int,
             pe_name: str = None):
    '''
    Method to run band structure calculations.

    Args:
        mater (str): Material name.
        kmesh (int): Number of k-points per line.
        nbands (int): Number of bands.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    scf = 'scf'
    bands = 'bands'
    scfdir = materdir / scf
    vaspdir = materdir / bands
    # Make vaspdir if not exist.
    __mkdir(vaspdir = vaspdir)
    incarposcar_scf = vasp.IncarPoscar.from_file(vaspdir = scfdir, calculation = scf)
    incarposcar = vasp.IncarPoscar.from_structure(structure = incarposcar_scf.structure, calculation = bands,
                                                  lsorbit = incarposcar_scf.lsorbit, nbands = nbands, vdw = incarposcar_scf.vdw)
    # Write INCAR.
    incarposcar.write_incar(vaspdir = vaspdir)
    # Write KPOINTS.
    incarposcar.write_kpoints(vaspdir = vaspdir, kmesh = kmesh)
    # Copy vdw_kernel.bindat if necessary.
    incarposcar.copy_vdw(vaspdir = vaspdir)
    jobscript = incarposcar.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write vasp.sh
    jobscript.write_file(dir = vaspdir)
    # Copy CHGCAR, POSCAR, and POTCAR.
    __copy_chgcar_poscar_potcar(scfdir = scfdir, vaspdir = vaspdir)
    # >> qsub vasp.sh
    subprocess.run(args = ['qsub', 'vasp.sh'], cwd = vaspdir)

def savebands(mater: str,
              with_wannier90: bool = False,
              ylim: list[float] = None):
    '''
    Method to save image of band structure plot.

    Args:
        mater (str): Material name.
        with_wannier90 (bool, optional): With or without Wannier90 band structure. Defaults to False.
        ylim (list[float], optional): Ylim [eV] of each Axes object. Length is 2. Defaults to None.
    '''
    materdir = HOME / __PACKAGE / mater
    efermi = vasp.get_efermi(vaspdir = materdir / 'scf')
    bandplot = vasp.get_bandplotter(banddir = materdir / 'bands', efermi = efermi)
    if with_wannier90:
        band = wannier90.BandDat.from_file(wanndir = materdir / 'nscf', seedname = 'wannier90')
        bandstructure = band.get_bandstructure(efermi = efermi)
        bandplot.add_bandstructure(bandstructure = bandstructure)
    bandplot.save_plot(filename = materdir / 'band.jpg', ylim = ylim)

def runnscf(mater: str,
            kmesh: list[int],
            nbands: int,
            atomic_orbitals: list[list[str]],
            queue_name: str,
            num_procs: int,
            exclude_bands: list[int] = None,
            pe_name: str = None):
    '''
    Method to run band structure calculations.

    Args:
        mater (str): Material name.
        kmesh (list[int]): Number of k-points for Monkhorst-Pack grids. Length is 3.
        nbands (int): Number of bands.
        atomic_orbitals (list[list[str]]): Projections. Length is num_atoms.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        exclude_bands (list[int], optional): Defaults to None.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    scf = 'scf'
    nscf = 'nscf'
    scfdir = materdir / scf
    vaspdir = materdir / nscf
    # Make vaspdir if not exist.
    __mkdir(vaspdir = vaspdir)
    incarposcar_scf = vasp.IncarPoscar.from_file(vaspdir = scfdir, calculation = scf)
    projections = wannier90.Projections.from_structure(incarposcar_scf.structure, *atomic_orbitals)
    exclude_bands = wannier90.SpecifyBands('exclude_bands', *exclude_bands) if exclude_bands is not None else None
    incarposcar = vasp.IncarPoscar.from_structure(structure = incarposcar_scf.structure, calculation = nscf,
                                                  lsorbit = incarposcar_scf.lsorbit, nbands = nbands, vdw = incarposcar_scf.vdw,
                                                  projections = projections, exclude_bands = exclude_bands)
    # Write INCAR.
    incarposcar.write_incar(vaspdir = vaspdir)
    # Write KPOINTS.
    incarposcar.write_kpoints(vaspdir = vaspdir, kmesh = kmesh)
    # Copy vdw_kernel.bindat if necessary.
    incarposcar.copy_vdw(vaspdir = vaspdir)
    jobscript = incarposcar.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write vasp.sh
    jobscript.write_file(dir = vaspdir)
    # Copy CHGCAR, POSCAR, and POTCAR.
    __copy_chgcar_poscar_potcar(scfdir = scfdir, vaspdir = vaspdir)
    # >> qsub vasp.sh
    subprocess.run(args = ['qsub', 'vasp.sh'], cwd = vaspdir)

def runw90(mater: str,
           bands_num_points: int,
           num_iter: int,
           queue_name: str,
           num_procs: int,
           dis_win_max: float = None,
           dis_froz_max: float = None,
           pe_name: str = None):
    '''
    Method to run Wannier90 calculations.

    Args:
        mater (str): Material name.
        bands_num_points (int): Number of k-points per line.
        num_iter (int): Number of wannierise iterations.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        dis_win_max (float, optional): Upper bound of outer window [eV] measured from pristine Fermi energy. Defaults to None.
        dis_froz_max (float, optional): Upper bound of inner window [eV] measured from pristine Fermi energy. Defaults to None.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    wanndir = materdir / 'nscf'
    seedname = 'wannier90'
    eig = wannier90.Eig.from_file(wanndir = wanndir, seedname = seedname)
    efermi = vasp.get_efermi(vaspdir = materdir / 'scf')
    win = wannier90.Win.from_file(wanndir = wanndir, seedname = seedname)
    win.append(eig = eig, efermi = efermi, bands_num_points = bands_num_points, num_iter = num_iter,
               dis_win_max = dis_win_max, dis_froz_max = dis_froz_max)
    # Write wannier90.win.
    win.write_file(wanndir = wanndir, seedname = seedname)
    jobscript = wannier90.get_jobscript(seedname = seedname, queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write wannier90.sh.
    jobscript.write_file(dir = wanndir)
    # >> qsub wannier90.sh
    subprocess.run(args = ['qsub', 'wannier90.sh'], cwd = wanndir)

def postw90_(mater: str,
             fermi_energy_range: list[float],
             kmesh: list[int],
             component: common.Component,
             queue_name: str,
             num_procs: int,
             pe_name: str = None):
    '''
    Method to run postw90 calculations.

    Args:
        mater (str): Material name.
        fermi_energy_range (list[float]): Range of Fermi energy [eV] measured from pristine Fermi energy.
        kmesh (list[int]): Number of k-points for berry and gyrotropic calculations. Length is 3.
        component (common.Component): Of spin Hall conductivity.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    wanndir = materdir / 'nscf'
    seedname = 'wannier90'
    efermi = vasp.get_efermi(vaspdir = materdir / 'scf')
    win = wannier90.Win.from_file(wanndir = wanndir, seedname = seedname)
    win.append_spin_hall(berry = True, gyrotropic = True,
                         efermi = efermi, fermi_energy_range = fermi_energy_range, kmesh = kmesh, component = component)
    # Write wannier90.win.
    win.write_file(wanndir = wanndir, seedname = seedname)
    # Write wannier90.spn if not exist.
    __write_spn(wanndir = wanndir)
    # Write wannier90.sHu if not exist.
    __write_uHu(wanndir = wanndir, extension = 'sHu')
    # Write wannier90.sIu if not exist.
    __write_uHu(wanndir = wanndir, extension = 'sIu')
    # Write wannier90.uHu if not exist.
    __write_uHu(wanndir = wanndir, extension = 'uHu')
    jobscript = postw90.get_jobscript(seedname = seedname, queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write postw90.sh.
    jobscript.write_file(dir = wanndir)
    # >> qsub postw90.sh
    subprocess.run(args = ['qsub', 'postw90.sh'], cwd = wanndir)

def savew90(mater: str,
            xlims: dict[common.Component, list[float]] = None,
            ylabels: dict[common.Component, str] = None,
            ylims: dict[common.Component, list[float]] = None):
    '''
    Method to save image of Fermi-energy dependence plot.

    Args:
        mater (str): Material name.
        xlims (dict[common.Component, list[float], optional): Xlim of each Axes object. Length is 2. Defaults to None.
        ylabels (dict[common.Component, str]): Ylabel of each Axes object. Defaults to None.
        ylims (dict[common.Component, list[float]], optional): Ylim of each Axes object. Length is 2. Defaults to None.
    '''
    materdir = HOME / __PACKAGE / mater
    wanndir = materdir / 'nscf'
    seedname = 'wannier90'
    efermi = vasp.get_efermi(vaspdir = materdir / 'scf')
    win = wannier90.Win.from_file(wanndir = wanndir, seedname = seedname)
    point_group = symmetry.PointGroup.from_structure(structure = win.structure)
    weights_acc_spin = point_group.get_response_coefficient(measure_axial = [True], force_axial = [False, False]).get_weights()
    gyrotropic = postw90.GyrotropicDat.acc_spin_from_file(wanndir = wanndir, seedname = seedname)
    acc_spins = [gyrotropic.get_physicalquantity_w_errors(efermi = efermi, weights = weights,
                                                          label = f'${list(weights)[0].to_xyz}$')
                 for weights in weights_acc_spin]
    physplot = common.PhysicalPlotter(acc_spins)
    physplot.save_plot(filename = materdir / 'she.jpg', xlims = xlims, ylabels = ylabels, ylims = ylims)

def runvasp():
    '''
    Method to run VASP calculations.
    '''
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('-y','--yaml', type = argparse.FileType('r'), required = True, help = 'Input yaml file')
    parser.add_argument('-m', '--mode',
                        choices = ['runscf', 'runbands', 'savebands', 'runnscf',
                                   'runw90', 'savebandsw90', 'postw90', 'savew90'],
                        required = True, help = 'Mode')
    args = parser.parse_args()
    kwargs = yaml.safe_load(stream = args.yaml)
    if args.mode == 'runscf':
        runscf(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'runbands':
        runbands(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'savebands':
        savebands(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'runnscf':
        runnscf(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'runw90':
        runw90(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'savebandsw90':
        savebands(mater = kwargs['mater'], with_wannier90 = True, **kwargs['savebands'])
    elif args.mode == 'postw90':
        postw90_(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'savew90':
        savew90(mater = kwargs['mater'], **kwargs[args.mode])

if __name__ == '__main__':
    runvasp()
