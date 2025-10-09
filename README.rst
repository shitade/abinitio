=========
abinitio
=========

Code for *ab initio* calculations
----------------------------------------------------------
This code is to deal with input/output files for VASP_, `Quantum Espresso`_, and Wannier90_.

.. _VASP: https://www.vasp.at/
.. _Quantum Espresso: https://www.quantum-espresso.org/
.. _Wannier90: https://wannier.org/

How to use
++++++++++
#. See __init__.py for preparation.

#. Prepare setting.yaml for your own environment.

#. Execute automatic/write_cif.py.

#. Prepare a yaml file for calculations. See automatic/Pt(W).yaml.

#. Execute automatic/runvasp(espresso).py.

#. You can generate a tex file for response coefficients allowed by (magnetic) point group symmetry using symmetry(magnetic).py.

Author
++++++

`Atsuo Shitade`_ (University of Osaka, JP)

.. _Atsuo Shitade: https://sites.google.com/view/shitade/