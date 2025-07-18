#!/usr/bin/env python
import numpy as np
'''
PySCF doesn't have its own input parser.  The input file is a Python program.

Before going throught the rest part, be sure the PySCF path is added in PYTHONPATH.
'''

import pyscf

# mol is an object to hold molecule information.
mol = pyscf.M(
    verbose = 4,
    output = 'out_h2o',
    atom = '''
    h 1.2300000 0.00000 0.000000
    h 1.2300000 1.230000 0.000000
    h 0.000000 1.23000000 0.000000 
    h 0.00000 0.00000 0.000000''',
    basis = 'sto-3g',
)
# For more details, see pyscf/gto/mole.py and pyscf/examples/gto

#
# The package follow the convention that each method has its class to hold
# control parameters.  The calculation can be executed by the kernel function.
# Eg, to do Hartree-Fock, (1) create HF object, (2) call kernel function
#
mf = mol.RHF()
print('E(HF)=%.15g' % mf.kernel())


#
# A post-HF method can be applied.
#

mp2 = mf.MP2().run()
print('E(MP2)=%.15g' % mp2.e_tot)

cc = mf.CCSD().run()
print('E(CCSD)=%.15g' % cc.e_tot)

print(8.987 * (4/1.23 + np.sqrt(2)/1.23) * 1.6 /27.21)
# More examples of pyscf input can be found in
# gto/00-input_mole.py
# gto/01-input_geometry.py
# gto/04-input_basis.py
# gto/05-input_ecp.py
# gto/06-load_mol_from_chkfile.py
# 1-advanced/002-input_script.py
