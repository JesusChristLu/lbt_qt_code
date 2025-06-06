#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CISD calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'H 1.300000 2.250000 0.000000;H 3.900000 2.250000 0.000000;H 5.200000 0.000000 0.000000;H 3.900000 -2.250000 0.000000;H 1.300000 -2.250000 0.000000 ;H 0.000000 0.000000 0.000000',
    basis = 'sto-3g')

mf = mol.HF().run()
mycc = mf.CISD().run()
print('RCISD correlation energy', mycc.e_corr)

mf = mol.UHF().run()
mycc = mf.CISD().run()
print('UCISD correlation energy', mycc.e_corr)

