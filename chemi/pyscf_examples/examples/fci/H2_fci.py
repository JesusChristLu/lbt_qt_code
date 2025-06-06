#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run FCI
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0.00000 0.0000 0.000000;H 0.00000 0.0000 0.74000000',
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



