from pyscf import gto, scf, fci
import numpy as np
# 使用经典计算化学软件PySCF的FCI方法来计算氢分子在不同键长下的能量
pyscf_energies = []
distances = np.linspace(0.18, 3.5, 50)
for bond_length in distances:
    # atom = f'H 0 0 0; Li 0 0 {bond_length}'
    # atom = f'N 0 0 0; N 0 0 {bond_length}'
    atom = f"H {bond_length} {bond_length} {bond_length}; H {-bond_length} {-bond_length} {bond_length}; H {-bond_length} {bond_length} {-bond_length}; H {bond_length} {-bond_length} {-bond_length}"

    mol = gto.M(atom=atom,   # in Angstrom
            basis='STO-3G',
            charge=1,
            spin=1)
    myhf = scf.HF(mol).run()
    cisolver = fci.FCI(myhf)
    pyscf_energies += [cisolver.kernel()[0]]

print(pyscf_energies)