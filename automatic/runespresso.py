'''
Module to run Quantum Espresso, Wannier90, and postw90.

Usage:
    >> python3 <Path/to/runespresso.py> -y <YAML> -m <MODE>
'''

from abinitio import HOME, common, espresso, wannier90, postw90, symmetry

__PACKAGE = 'espresso'

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
    
    Note:
        vdw is not implemented for Quantum Espresso.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    structure = common.get_structure_from_cif(dir = materdir, mater = mater, magmom = magmom)
    inputpw = espresso.InputPW.from_structure(structure = structure, calculation = 'scf', lspinorb = soc, kmesh = kmesh)
    # Write pwscf.in.
    inputpw.write_file(qedir = materdir)
    jobscript = inputpw.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write pwscf.sh.
    jobscript.write_file(dir = materdir)
    # >> qsub pwscf.sh
    subprocess.run(args = ['qsub', 'pwscf.sh'], cwd = materdir)

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
    inputpw_scf = espresso.InputPW.from_file(filename = materdir / 'pwscf.in')
    inputpw = espresso.InputPW.from_structure(structure = inputpw_scf.structure, calculation = 'bands',
                                              lspinorb = inputpw_scf.system.lspinorb, kmesh = kmesh, nbnd = nbands)
    # Write pwbands.in.
    inputpw.write_file(qedir = materdir)
    jobscript = inputpw.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write pwbands.sh.
    jobscript.write_file(dir = materdir)
    # >> qsub pwbands.sh
    subprocess.run(args = ['qsub', 'pwbands.sh'], cwd = materdir)

def postbands(mater: str,
              queue_name: str,
              num_procs: int,
              pe_name: str = None):
    '''
    Method to run post-process of band structure calculations.

    Args:
        mater (str): Material name.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    inputpw_bands = espresso.InputPW.from_file(filename = materdir / 'pwbands.in')
    bands = inputpw_bands.get_bands()
    # Write bands.in.
    bands.write_file(qedir = materdir)
    jobscript = bands.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write bands.sh.
    jobscript.write_file(dir = materdir)
    # >> qsub bands.sh
    subprocess.run(args = ['qsub', 'bands.sh'], cwd = materdir)

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
    inputpw_bands = espresso.InputPW.from_file(filename = materdir / 'pwbands.in')
    efermi = espresso.get_efermi(qedir = materdir)
    gnu = espresso.BandsGnu.from_file(structure = inputpw_bands.structure, qedir = materdir)
    bandplot = gnu.get_bandplotter(structure = inputpw_bands.structure, efermi = efermi)
    if with_wannier90:
        band = wannier90.BandDat.from_file(wanndir = materdir, seedname = 'wannier90')
        bandstructure = band.get_bandstructure(efermi = efermi)
        bandplot.add_bandstructure(bandstructure = bandstructure)
    bandplot.save_plot(filename = materdir / 'band.jpg', ylim = ylim)

def runnscf(mater: str,
            kmesh: int,
            nbands: int,
            queue_name: str,
            num_procs: int,
            pe_name: str = None):
    '''
    Method to run non-self-consisntent calculations.

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
    inputpw_scf = espresso.InputPW.from_file(filename = materdir / 'pwscf.in')
    inputpw = espresso.InputPW.from_structure(structure = inputpw_scf.structure, calculation = 'nscf',
                                              lspinorb = inputpw_scf.system.lspinorb, kmesh = kmesh, nbnd = nbands)
    # Write pwnscf.in.
    inputpw.write_file(qedir = materdir)
    jobscript = inputpw.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write pwnscf.sh.
    jobscript.write_file(dir = materdir)
    # >> qsub pwnscf.sh
    subprocess.run(args = ['qsub', 'pwnscf.sh'], cwd = materdir)

def prew90(mater: str,
           atomic_orbitals: list[list[str]],
           queue_name: str,
           num_procs: int,
           exclude_bands: list[int] = None,
           pe_name: str = None):
    '''
    Method to run pre-process of Wannier90 calculations.

    Args:
        mater (str): Material name.
        atomic_orbitals (list[list[str]]): Projections. Length is num_atoms.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        exclude_bands (list[int], optional): Defaults to None.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    inputpw_nscf = espresso.InputPW.from_file(filename = materdir / 'pwnscf.in')
    projections = wannier90.Projections.from_structure(inputpw_nscf.structure, *atomic_orbitals)
    exclude_bands = wannier90.SpecifyBands('exclude_bands', *exclude_bands) if exclude_bands is not None else None
    win = inputpw_nscf.get_win(projections = projections, exclude_bands = exclude_bands)
    # Write wannier90.win.
    win.write_file(wanndir = materdir, seedname = 'wannier90')
    # Write wannier90.sh to get wannier90.nnkp.
    jobscript = wannier90.get_jobscript(seedname = '-pp wannier90', queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    jobscript.job_name = 'wannier90pp'
    jobscript.write_file(dir = materdir)
    # >> qsub wannier90pp.sh
    subprocess.run(args = ['qsub', 'wannier90pp.sh'], cwd = materdir)

def pw2w90(mater: str,
           queue_name: str,
           num_procs: int,
           pe_name: str = None):
    '''
    Method to run pw2wannier90 calculations.

    Args:
        mater (str): Material name.
        queue_name (str): Queue name.
        num_procs (int): Number of processors.
        pe_name (str, optional): Name of parallel environment. Defaults to None, leading to default value in configuration file.
    '''
    import subprocess
    materdir = HOME / __PACKAGE / mater
    inputpw_nscf = espresso.InputPW.from_file(filename = materdir / 'pwnscf.in')
    # Write pw2wannier90.in.
    inputpp = inputpw_nscf.get_inputpp(shc = True, acc_spin = True)
    inputpp.write_file(qedir = materdir)
    # Write pw2wannier90.sh
    jobscript = inputpp.get_jobscript(queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    jobscript.write_file(dir = materdir)
    # >> qsub pw2wannier90.sh
    subprocess.run(args = ['qsub', 'pw2wannier90.sh'], cwd = materdir)

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
    seedname = 'wannier90'
    eig = wannier90.Eig.from_file(wanndir = materdir, seedname = seedname)
    efermi = espresso.get_efermi(qedir = materdir)
    win = wannier90.Win.from_file(wanndir = materdir, seedname = seedname)
    win.append(eig = eig, efermi = efermi, bands_num_points = bands_num_points, num_iter = num_iter,
               dis_win_max = dis_win_max, dis_froz_max = dis_froz_max)
    # Write wannier90.win.
    win.write_file(wanndir = materdir, seedname = seedname)
    jobscript = wannier90.get_jobscript(seedname = seedname, queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write wannier90.sh.
    jobscript.write_file(dir = materdir)
    # >> qsub wannier90.sh
    subprocess.run(args = ['qsub', 'wannier90.sh'], cwd = materdir)

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
    seedname = 'wannier90'
    efermi = espresso.get_efermi(qedir = materdir)
    win = wannier90.Win.from_file(wanndir = materdir, seedname = seedname)
    win.append_spin_hall(berry = True, gyrotropic = True,
                         efermi = efermi, fermi_energy_range = fermi_energy_range, kmesh = kmesh, component = component)
    # Write wannier90.win.
    win.write_file(wanndir = materdir, seedname = seedname)
    jobscript = postw90.get_jobscript(seedname = seedname, queue_name = queue_name, num_procs = num_procs, pe_name = pe_name)
    # Write postw90.sh.
    jobscript.write_file(dir = materdir)
    # >> qsub postw90.sh
    subprocess.run(args = ['qsub', 'postw90.sh'], cwd = materdir)

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
    seedname = 'wannier90'
    efermi = espresso.get_efermi(qedir = materdir)
    win = wannier90.Win.from_file(wanndir = materdir, seedname = seedname)
    point_group = symmetry.PointGroup.from_structure(structure = win.structure)
    weights_acc_spin = point_group.get_response_coefficient(rank = 3, is_axial = True).get_weights()
    gyrotropic = postw90.GyrotropicDat.acc_spin_from_file(wanndir = materdir, seedname = seedname)
    acc_spins = [gyrotropic.get_physicalquantity_w_errors(efermi = efermi, weights = weights,
                                                          label = f'${list(weights)[0].to_xyz}$')
                 for weights in weights_acc_spin]
    physplot = common.PhysicalPlotter(acc_spins)
    physplot.save_plot(filename = materdir / 'she.jpg', xlims = xlims, ylabels = ylabels, ylims = ylims)

def runespresso():
    '''
    Method to run Quantum Espresso calculations.
    '''
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('-y','--yaml', type = argparse.FileType('r'), required = True, help = 'Input yaml file')
    parser.add_argument('-m', '--mode',
                        choices = ['runscf', 'runbands', 'postbands', 'savebands', 'runnscf', 'prew90', 'pw2w90',
                                   'runw90', 'savebandsw90', 'postw90', 'savew90'],
                        required = True, help = 'Mode')
    args = parser.parse_args()
    kwargs = yaml.safe_load(stream = args.yaml)
    if args.mode == 'runscf':
        runscf(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'runbands':
        runbands(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'postbands':
        postbands(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'savebands':
        savebands(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'runnscf':
        kwargs[args.mode].pop('atomic_orbitals', None)
        kwargs[args.mode].pop('exclude_bands', None)
        runnscf(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'prew90':
        prew90(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'pw2w90':
        pw2w90(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'runw90':
        runw90(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'savebandsw90':
        savebands(mater = kwargs['mater'], with_wannier90 = True, **kwargs['savebands'])
    elif args.mode == 'postw90':
        postw90_(mater = kwargs['mater'], **kwargs[args.mode])
    elif args.mode == 'savew90':
        savew90(mater = kwargs['mater'], **kwargs[args.mode])

if __name__ == '__main__':
    runespresso()
