#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run FCI
'''

import pyscf

mol = pyscf.M(
    atom = 'H 1.300000 2.250000 0.000000;H 3.900000 2.250000 0.000000',
    basis = 'sto-3g',
    symmetry = True,
)
myhf = mol.RHF().run()

#
# create an FCI solver based on the SCF object
#
cisolver = pyscf.fci.FCI(myhf)
print('E(FCI) = %.12f' % cisolver.kernel()[0])

#
# create an FCI solver based on the SCF object
#
myuhf = mol.UHF().run()
cisolver = pyscf.fci.FCI(myuhf)
print('E(UHF-FCI) = %.12f' % cisolver.kernel()[0])

#
# create an FCI solver based on the given orbitals and the num. electrons and
# spin of the mol object
#
cisolver = pyscf.fci.FCI(mol, myhf.mo_coeff)
print('E(FCI) = %.12f' % cisolver.kernel()[0])



