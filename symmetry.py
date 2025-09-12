'''
Module to compute response coefficients allowed by point group symmetry.
Save symmetry.tex for all point groups if executed.

Usage:
    >> python3 <Path/to/symmetry.py>
'''

import numpy as np
from pathlib import Path
from pymatgen.core.structure import Structure
import sympy as sp
from abinitio import HOME, common

class ResponseCoefficient:
    '''
    Class of response coefficients.

    Args:
        rank (int): Rank.
        is_axial (bool): Odd number of axial vectors are involved or not.
        is_time_reversal_odd (bool, optional): Odd number of time-reversal-odd vectors are involved or not. Defaults to False.
        tensor (sp.ImmutableDenseNDimArray, optional): With shape (3,) * rank. Defaults to None.

    Attributes:
        latex (str): LaTeX document without pre/postamble.
    '''
    def __init__(self,
                 rank: int,
                 is_axial: bool,
                 is_time_reversal_odd: bool = False,
                 tensor: sp.ImmutableDenseNDimArray = None):
        self.rank = rank
        self.is_axial = is_axial
        self.is_time_reversal_odd = is_time_reversal_odd
        if tensor is not None:
            self.tensor = tensor
        else:
            self.tensor = sp.ImmutableDenseNDimArray(sp.symbols(names = 'a_{' + '(x:z)' * rank + '}')).reshape(*(3,) * rank)
    def __str__(self):
        polar_or_axial = 'polar' if not self.is_axial else 'axial'
        odd_or_even = 'time-reversal-even' if not self.is_time_reversal_odd else 'time-reversal-odd'
        lines = [f'Rank-{self.rank} {polar_or_axial} {odd_or_even} {__class__.__name__}',
                 str(self.tensor)]
        return '\n'.join(lines)
    @property
    def latex(self)-> str:
        '''
        LaTeX document without pre/postamble.
        '''
        polar_or_axial = 'polar' if not self.is_axial else 'axial'
        odd_or_even = 'time-reversal-even' if not self.is_time_reversal_odd else 'time-reversal-odd'
        lines = [f'Rank-${self.rank}$ {polar_or_axial} {odd_or_even} {__class__.__name__}',
                 r'  \[',
                 f'    {sp.latex(self.tensor)}.',
                 r'  \]']
        return '\n'.join(lines)
    def get_weights(self)-> list[dict[common.Component, float]]:
        '''
        Method to get weights of symmetry-allowed components.

        Returns:
            list[dict[common.Component, float]]: Weights of symmetry-allowed components.
        '''
        symbols = sp.symbols('a_{' + '(x:z)' * self.rank + '}')
        weights = []
        for nonzero_symbol in symbols:
            # Substitute symbol = 1 for nonzero_symbol,
            #                   = 0 else.
            tensor = self.tensor.subs([(symbol, 1 if symbol == nonzero_symbol else 0)
                                       for symbol in symbols])
            nonzero_components = list(zip(*np.nonzero(a = tensor)))
            if nonzero_components:
                weight = {common.Component.from_tuple(*component): tensor[component]
                          for component in nonzero_components}
                weights.append(weight)
        return weights

class SymmOp:
    '''
    Class of rotation parts of symmetry operations.

    Args:
        is_hexagonal (bool): Symmetry operation in hexagonal lattice or not.
        seitz (str): Seitz symbol.
        is_time_reversal (bool): Symmetry operation involves time-reversal operation or not.

    Attributes:
        latex (str): LaTeX document without pre/postamble.
        det (int): Determinant of matrix form.
        matrix_conv (sp.MutableDenseMatrix): Matrix form in conventional lattice.
        matrix_prim (sp.MutableDenseMatrix): Matrix form in primitive lattice.
        triplet (list[sp.Symbol]): Coordinate triplet form in primitive lattice.
    '''
    x = np.array(object = [1, 0, 0], dtype = int)
    y = np.array(object = [0, 1, 0], dtype = int)
    z = np.array(object = [0, 0, 1], dtype = int)
    __r = sp.MutableDenseMatrix(sp.symbols('x:z'))
    __avecs_hexagonal = sp.MutableDenseMatrix([[1, sp.cos(2 * sp.pi / 3), 0],
                                               [0, sp.sin(2 * sp.pi / 3), 0],
                                               [0, 0, 1]])
    # Seitz symbols and coordinate triplet forms of symmetry operations
    # in cubic, tetragonal, orthorhombic, monoclinic and triclinic crystals if False,
    # in hexagonal and trigonal crystals if True.
    ROTATIONS = {
        False: {
            '1': [x, y, z],
            '2_{001}': [-x, -y, z],
            '2_{010}': [-x, y, -z],
            '2_{100}': [x, -y, -z],
            '3^{+}_{111}': [z, x, y],
            '3^{+}_{-11-1}': [z, -x, -y],
            '3^{+}_{1-1-1}': [-z, -x, y],
            '3^{+}_{-1-11}': [-z, x, -y],
            '3^{-}_{111}': [y, z, x],
            '3^{-}_{1-1-1}': [-y, z, -x],
            '3^{-}_{-1-11}': [y, -z, -x],
            '3^{-}_{-11-1}': [-y, -z, x],
            '2_{110}': [y, x, -z],
            '2_{1-10}': [-y, -x, -z],
            '4^{-}_{001}': [y, -x, z],
            '4^{+}_{001}': [-y, x, z],
            '4^{-}_{100}': [x, z, -y],
            '2_{011}': [-x, z, y],
            '2_{01-1}': [-x, -z, -y],
            '4^{+}_{100}': [x, -z, y],
            '4^{+}_{010}': [z, y, -x],
            '2_{101}': [z, -y, x],
            '4^{-}_{010}': [-z, y, x],
            '2_{-101}': [-z, -y, -x],
            '-1': [-x, -y, -z],
            'm_{001}': [x, y, -z],
            'm_{010}': [x, -y, z],
            'm_{100}': [-x, y, z],
            '-3^{+}_{111}': [-z, -x, -y],
            '-3^{+}_{-11-1}': [-z, x, y],
            '-3^{+}_{1-1-1}': [z, x, -y],
            '-3^{+}_{-1-11}': [z, -x, y],
            '-3^{-}_{111}': [-y, -z, -x],
            '-3^{-}_{1-1-1}': [y, -z, x],
            '-3^{-}_{-1-11}': [-y, z, x],
            '-3^{-}_{-11-1}': [y, z, -x],
            'm_{110}': [-y, -x, z],
            'm_{1-10}': [y, x, z],
            '-4^{-}_{001}': [-y, x, -z],
            '-4^{+}_{001}': [y, -x, -z],
            '-4^{-}_{100}': [-x, -z, y],
            'm_{011}': [x, -z, -y],
            'm_{01-1}': [x, z, y],
            '-4^{+}_{100}': [-x, z, -y],
            '-4^{+}_{010}': [-z, -y, x],
            'm_{101}': [-z, y, -x],
            '-4^{-}_{010}': [z, -y, -x],
            'm_{-101}': [z, y, x]
        },
        True: {
            '1': [x, y, z],
            '3^{+}_{001}': [-y, x - y, z],
            '3^{-}_{001}': [-x + y, -x, z],
            '2_{001}': [-x, -y, z],
            '6^{-}_{001}': [y, -x + y, z],
            '6^{+}_{001}': [x - y, x, z],
            '2_{110}': [y, x, -z],
            '2_{100}': [x - y, -y, -z],
            '2_{010}': [-x, -x + y, -z],
            '2_{1-10}': [-y, -x, -z],
            '2_{120}': [-x + y, y, -z],
            '2_{210}': [x, x - y, -z],
            '-1': [-x, -y, -z],
            '-3^{+}_{001}': [y, -x + y, -z],
            '-3^{-}_{001}': [x - y, x, -z],
            'm_{001}': [x, y, -z],
            '-6^{-}_{001}': [-y, x - y, -z],
            '-6^{+}_{001}': [-x + y, -x, -z],
            'm_{110}': [-y, -x, z],
            'm_{100}': [-x + y, y, z],
            'm_{010}': [x, x - y, z],
            'm_{1-10}': [y, x, z],
            'm_{120}': [x - y, -y, z],
            'm_{210}': [-x, -x + y, z]
        }
    }
    def __init__(self,
                 is_hexagonal: bool,
                 seitz: str):
        self.is_hexagonal = is_hexagonal
        self.seitz = seitz.strip("'")
        self.is_time_reversal = seitz.endswith("'")
    def __str__(self):
        # Time-reversal part = +1.
        if not self.is_time_reversal:
            return f"{self.triplet} +1 (Seitz symbol = {self.seitz}, det = {self.det})"
        # Time-reversal part = -1.
        else:
            return f"{self.triplet} -1 (Seitz symbol = {self.seitz}', det = {self.det})"
    @property
    def latex(self):
        '''
        LaTeX document without pre/postamble.
        '''
        # Time-reversal part = +1.
        if not self.is_time_reversal:
            return fr"${self.triplet}$ $+1$ (Seitz symbol = ${self.seitz}$, $\det = {self.det}$)"
        # Time-reversal part = -1.
        else:
            return fr"${self.triplet}$ $-1$ (Seitz symbol = ${{{self.seitz}}}'$, $\det = {self.det}$)"
    @property
    def det(self)-> int:
        '''
        Determinant of matrix form.
        '''
        return int(self.matrix_prim.det())
    @property
    def matrix_conv(self)-> sp.MutableDenseMatrix:
        '''
        Matrix form in conventional lattice.
        '''
        if not self.is_hexagonal:
            return self.matrix_prim
        else:
            return self.__avecs_hexagonal * self.matrix_prim * self.__avecs_hexagonal.inv()
    @property
    def matrix_prim(self)-> sp.MutableDenseMatrix:
        '''
        Matrix form in primitive lattice.
        '''
        return sp.MutableDenseMatrix(self.ROTATIONS[self.is_hexagonal][self.seitz])
    @property
    def triplet(self)-> list[sp.Symbol]:
        '''
        Coordinate triplet form in primitive lattice.
        '''
        return list(self.matrix_prim * __class__.__r)

class PointGroup:
    '''
    Class of point groups.

    Args:
        hermann_mauguin (str): Hermann-Mauguin symbol.

    Attributes:
        latex (str): LaTeX document without pre/postamble.
        is_centrosymmetric (bool): Centrosymmetric or not.
        is_chiral (bool): Chiral or not.
        is_hexagonal (bool): Hexagonal or not.
        order (int): Order, the number of symmetry operations.
        symmops (list[SymmOp]): Symmetry operations.
    '''
    ROTATIONS = {
        # Triclinic.
        '1': ['1'],
        '-1': ['1',
               '-1'],
        # Monoclinic.
        '2': ['1', '2_{001}'],
        'm': ['1', 'm_{001}'],
        '2/m': ['1', '2_{001}',
                '-1', 'm_{001}'],
        # Orthorhombic.
        '222': ['1', '2_{001}', '2_{010}', '2_{100}'],
        'mm2': ['1', '2_{001}', 'm_{010}', 'm_{100}'],
        'mmm': ['1', '2_{001}', '2_{010}', '2_{100}',
                '-1', 'm_{001}', 'm_{010}', 'm_{100}'],
        # Tetragonal.
        '4': ['1', '2_{001}', '4^{+}_{001}', '4^{-}_{001}'],
        '-4': ['1', '2_{001}', '-4^{+}_{001}', '-4^{-}_{001}'],
        '4/m': ['1', '2_{001}', '4^{+}_{001}', '4^{-}_{001}',
                '-1', 'm_{001}', '-4^{+}_{001}', '-4^{-}_{001}'],
        '422': ['1', '2_{001}', '4^{+}_{001}', '4^{-}_{001}', '2_{010}', '2_{100}', '2_{110}', '2_{1-10}'],
        '4mm': ['1', '2_{001}', '4^{+}_{001}', '4^{-}_{001}', 'm_{010}', 'm_{100}', 'm_{110}', 'm_{1-10}'],
        '-42m': ['1', '2_{001}', '-4^{+}_{001}', '-4^{-}_{001}', '2_{010}', '2_{100}', 'm_{110}', 'm_{1-10}'],
        '4/mmm': ['1', '2_{001}', '4^{+}_{001}', '4^{-}_{001}', '2_{010}', '2_{100}', '2_{110}', '2_{1-10}',
                  '-1', 'm_{001}', '-4^{+}_{001}', '-4^{-}_{001}', 'm_{010}', 'm_{100}', 'm_{110}', 'm_{1-10}'],
        # Trigonal.
        '3': ['1', '3^{+}_{001}', '3^{-}_{001}'],
        '-3': ['1', '3^{+}_{001}', '3^{-}_{001}',
               '-1', '-3^{+}_{001}', '-3^{-}_{001}'],
        '32': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{110}', '2_{100}', '2_{010}'],
        '3m': ['1', '3^{+}_{001}', '3^{-}_{001}', 'm_{110}', 'm_{100}', 'm_{010}'],
        '-3m': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{110}', '2_{100}', '2_{010}',
                 '-1', '-3^{+}_{001}', '-3^{-}_{001}', 'm_{110}', 'm_{100}', 'm_{010}'],
        # Hexagonal.
        '6': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{001}', '6^{-}_{001}', '6^{+}_{001}'],
        '-6': ['1', '3^{+}_{001}', '3^{-}_{001}', 'm_{001}', '-6^{-}_{001}', '-6^{+}_{001}'],
        '6/m': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{001}', '6^{-}_{001}', '6^{+}_{001}',
                '-1', '-3^{+}_{001}', '-3^{-}_{001}', 'm_{001}', '-6^{-}_{001}', '-6^{+}_{001}'],
        '622': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{001}', '6^{-}_{001}', '6^{+}_{001}', '2_{110}', '2_{100}', '2_{010}', '2_{1-10}', '2_{120}', '2_{210}'],
        '6mm': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{001}', '6^{-}_{001}', '6^{+}_{001}', 'm_{110}', 'm_{100}', 'm_{010}', 'm_{1-10}', 'm_{120}', 'm_{210}'],
        '-6m2': ['1', '2_{1-10}', '2_{120}', '2_{210}', '3^{+}_{001}', '3^{-}_{001}', 'm_{001}', '-6^{-}_{001}', '-6^{+}_{001}', 'm_{110}', 'm_{100}', 'm_{010}'],
        '6/mmm': ['1', '3^{+}_{001}', '3^{-}_{001}', '2_{001}', '6^{-}_{001}', '6^{+}_{001}', '2_{110}', '2_{100}', '2_{010}', '2_{1-10}', '2_{120}', '2_{210}',
                  '-1', '-3^{+}_{001}', '-3^{-}_{001}', 'm_{001}', '-6^{-}_{001}', '-6^{+}_{001}', 'm_{110}', 'm_{100}', 'm_{010}', 'm_{1-10}', 'm_{120}', 'm_{210}'],
        # Cubic.
        '23': ['1', '2_{001}', '2_{010}', '2_{100}', '3^{+}_{111}', '3^{+}_{-11-1}', '3^{+}_{1-1-1}', '3^{+}_{-1-11}', '3^{-}_{111}', '3^{-}_{1-1-1}', '3^{-}_{-1-11}', '3^{-}_{-11-1}'],
        'm-3': ['1', '2_{001}', '2_{010}', '2_{100}', '3^{+}_{111}', '3^{+}_{-11-1}', '3^{+}_{1-1-1}', '3^{+}_{-1-11}', '3^{-}_{111}', '3^{-}_{1-1-1}', '3^{-}_{-1-11}', '3^{-}_{-11-1}',
                '-1', 'm_{001}', 'm_{010}', 'm_{100}', '-3^{+}_{111}', '-3^{+}_{-11-1}', '-3^{+}_{1-1-1}', '-3^{+}_{-1-11}', '-3^{-}_{111}', '-3^{-}_{1-1-1}', '-3^{-}_{-1-11}', '-3^{-}_{-11-1}'],
        '432': ['1', '2_{001}', '2_{010}', '2_{100}', '3^{+}_{111}', '3^{+}_{-11-1}', '3^{+}_{1-1-1}', '3^{+}_{-1-11}', '3^{-}_{111}', '3^{-}_{1-1-1}', '3^{-}_{-1-11}', '3^{-}_{-11-1}', '2_{110}', '2_{1-10}', '4^{-}_{001}', '4^{+}_{001}', '4^{-}_{100}', '2_{011}', '2_{01-1}', '4^{+}_{100}', '4^{+}_{010}', '2_{101}', '4^{-}_{010}', '2_{-101}'],
        '-43m': ['1', '2_{001}', '2_{010}', '2_{100}', '3^{+}_{111}', '3^{+}_{-11-1}', '3^{+}_{1-1-1}', '3^{+}_{-1-11}', '3^{-}_{111}', '3^{-}_{1-1-1}', '3^{-}_{-1-11}', '3^{-}_{-11-1}', 'm_{1-10}', 'm_{110}', '-4^{+}_{001}', '-4^{-}_{001}', 'm_{01-1}', '-4^{+}_{100}', '-4^{-}_{100}', 'm_{011}', 'm_{-101}', '-4^{-}_{010}', 'm_{101}', '-4^{+}_{010}'],
        'm-3m': ['1', '2_{001}', '2_{010}', '2_{100}', '3^{+}_{111}', '3^{+}_{-11-1}', '3^{+}_{1-1-1}', '3^{+}_{-1-11}', '3^{-}_{111}', '3^{-}_{1-1-1}', '3^{-}_{-1-11}', '3^{-}_{-11-1}', '2_{110}', '2_{1-10}', '4^{-}_{001}', '4^{+}_{001}', '4^{-}_{100}', '2_{011}', '2_{01-1}', '4^{+}_{100}', '4^{+}_{010}', '2_{101}', '4^{-}_{010}', '2_{-101}',
                 '-1', 'm_{001}', 'm_{010}', 'm_{100}', '-3^{+}_{111}', '-3^{+}_{-11-1}', '-3^{+}_{1-1-1}', '-3^{+}_{-1-11}', '-3^{-}_{111}', '-3^{-}_{1-1-1}', '-3^{-}_{-1-11}', '-3^{-}_{-11-1}', 'm_{110}', 'm_{1-10}', '-4^{-}_{001}', '-4^{+}_{001}', '-4^{-}_{100}', 'm_{011}', 'm_{01-1}', '-4^{+}_{100}', '-4^{+}_{010}', 'm_{101}', '-4^{-}_{010}', 'm_{-101}']
    }
    HEXAGONALS = ['3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm']
    def __init__(self,
                 hermann_mauguin: str):
        self.hermann_mauguin = hermann_mauguin
    def __str__(self):
        lines = ['------------------------------',
                 f'{__class__.__name__}: {self.hermann_mauguin}',
                 '------------------------------',
                 f'Order: {self.order}',
                 f'Inversion: {self.is_centrosymmetric}',
                 f'Chirality: {self.is_chiral}',
                 f'List of symmetry operations:']
        # 0-based index i to 1-based index i + 1.
        lines.extend(f'- {i + 1}: {symmop}' for (i, symmop) in enumerate(self.symmops))
        lines.extend([str(self.get_response_coefficient(rank = 1, is_axial = False)),
                      str(self.get_response_coefficient(rank = 1, is_axial = True)),
                      str(self.get_response_coefficient(rank = 2, is_axial = False)),
                      str(self.get_response_coefficient(rank = 2, is_axial = True)),
                      str(self.get_response_coefficient(rank = 3, is_axial = False)),
                      str(self.get_response_coefficient(rank = 3, is_axial = True))])
        return '\n'.join(lines)
    @property
    def latex(self)-> str:
        '''
        LaTeX document without pre/postamble.
        '''
        lines = [fr'\section{{{__class__.__name__}: ${self.hermann_mauguin}$}}',
                 r'\begin{itemize}',
                 fr'  \item Order: ${self.order}$.',
                 fr'  \item Inversion: {self.is_centrosymmetric}.',
                 fr'  \item Chirality: {self.is_chiral}.',
                 r'  \item List of symmetry operations:',
                 r'  \begin{enumerate}']
        lines.extend(fr'    \item {symmop.latex}' for symmop in self.symmops)
        lines.extend([r'  \end{enumerate}',
                      r'  \item ' + self.get_response_coefficient(rank = 1, is_axial = False).latex,
                      r'  \item ' + self.get_response_coefficient(rank = 1, is_axial = True).latex,
                      r'  \item ' + self.get_response_coefficient(rank = 2, is_axial = False).latex,
                      r'  \item ' + self.get_response_coefficient(rank = 2, is_axial = True).latex,
                      r'  \item ' + self.get_response_coefficient(rank = 3, is_axial = False).latex,
                      r'  \item ' + self.get_response_coefficient(rank = 3, is_axial = True).latex,
                      r'\end{itemize}',
                      ''])
        return '\n'.join(lines)
    @property
    def is_centrosymmetric(self)-> bool:
        '''
        Centrosymmetric or not.
        '''
        return '-1' in self.ROTATIONS[self.hermann_mauguin]
    @property
    def is_chiral(self)-> bool:
        '''
        Chiral or not.
        '''
        return all(symmop.det == 1
                   for symmop in self.symmops)
    @property
    def is_hexagonal(self)-> bool:
        '''
        Hexagonal or not.
        '''
        return self.hermann_mauguin in self.HEXAGONALS
    @property
    def order(self)-> int:
        '''
        Order, the number of symmetry operations.
        '''
        return len(self.ROTATIONS[self.hermann_mauguin])
    @property
    def symmops(self)-> list[SymmOp]:
        '''
        Symmetry operations.
        '''
        return [SymmOp(is_hexagonal = self.is_hexagonal, seitz = seitz)
                for seitz in self.ROTATIONS[self.hermann_mauguin]]
    @classmethod
    def from_structure(cls,
                       structure: Structure):
        '''
        Classmethod from Structure object.

        Args:
            structure (Structure):
        '''
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return cls(hermann_mauguin = SpacegroupAnalyzer(structure = structure).get_point_group_symbol())
    def check_closure(self)-> bool:
        '''
        Check if implemented point group is closed.

        Returns:
            bool: True for closed.
        '''
        symmmats = [symmop.matrix_prim
                    for symmop in self.symmops]
        for symmmat_i in symmmats:
            for symmmat_j in symmmats:
                symmmat_prod = symmmat_i * symmmat_j
                if not symmmat_prod in symmmats:
                    return False
        return True
    def get_response_coefficient(self,
                                 rank: int,
                                 is_axial: bool,
                                 is_time_reversal_odd: bool = False)-> ResponseCoefficient:
        '''
        Method to get symmetry-allowed ResponseCoefficient object.

        Args:
            rank (int): Rank.
            is_axial (bool): Odd number of axial vectors are involved or not.
            is_time_reversal_odd (bool, optional): Odd number of time-reversal-odd vectors are involved or not. Defaults to False.

        Returns:
            ResponseCoefficient: Allowed by symmetry.
        '''
        coeff = ResponseCoefficient(rank = rank, is_axial = is_axial, is_time_reversal_odd = is_time_reversal_odd)
        constraint = []
        for symmop in self.symmops:
            rotated_tensor = coeff.tensor
            for _ in range(coeff.rank):
                rotated_tensor = sp.tensorcontraction(sp.tensorproduct(symmop.matrix_conv, rotated_tensor), (1, coeff.rank + 1))
            # Multiply determinant if response is axial.
            if is_axial:
                rotated_tensor = symmop.det * rotated_tensor
            # Multipy -1 if response is time-reversal-odd and symmetry operation involves time-reversal operation.
            if is_time_reversal_odd and symmop.is_time_reversal:
                rotated_tensor = -rotated_tensor
            diff_tensor = rotated_tensor - coeff.tensor
            # Difference between rotated and original tensor.
            constraint.extend(diff_tensor.reshape(3 ** coeff.rank).tolist())
        # Reverse list of components to express solution with ascending constants (decending variables).
        solution = sp.solve(constraint, coeff.tensor.reshape(3 ** coeff.rank).tolist()[::-1])
        return ResponseCoefficient(rank = rank, is_axial = is_axial, is_time_reversal_odd = is_time_reversal_odd,
                                   tensor = coeff.tensor.subs(solution))
    def write_latex(self,
                    filename: Path):
        '''
        Method to write LaTeX file.

        Args:
            filename (Path):
        '''
        with open(file = filename, mode = 'a') as f:
            f.write(self.latex)
        print(f'## Appending {self.hermann_mauguin} to {filename} finished. ##')

def write_preamble(filename: Path):
    lines = [
        r'\documentclass{article}',
        r'\usepackage[top=20truemm,bottom=20truemm,left=20truemm,right=20truemm]{geometry}',
        r'\usepackage{amsmath,amssymb,txfonts,fontspec}',
        r'\usepackage[pdfencoding=auto]{hyperref}',
        r'\allowdisplaybreaks[4]',
        r'\begin{document}',
        r'\tableofcontents',
        ''
        ]
    with open(file = filename, mode = 'w') as f:
        f.write('\n'.join(lines))
    print(f'## Writing {filename} finished. ##')

def write_postamble(filename: Path):
    lines = [
        r'\end{document}',
        ''
        ]
    with open(file = filename, mode = 'a') as f:
        f.write('\n'.join(lines))
    print(f'## Appending to {filename} finished. ##')

def __write_latex(filename: Path):
    '''
    Method to write LaTeX file of all point groups.

    Args:
        filename (Path):
    '''
    write_preamble(filename = filename)
    for (i, hermann_mauguin) in enumerate(list(PointGroup.ROTATIONS.keys())):
        pg = PointGroup(hermann_mauguin = hermann_mauguin)
        pg.write_latex(filename = filename)
    write_postamble(filename = filename)

if __name__ == '__main__':
    __write_latex(filename = HOME / 'symmetry.tex')
