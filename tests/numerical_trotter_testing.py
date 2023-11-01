import unittest
import GKSHamSimModule as hs
import numpy as np
from qiskit.providers.aer.backends.statevector_simulator import StatevectorSimulator
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.providers.aer import QasmSimulator


qtester = QuantumCircuit(2)
qtester.h(0)
qtester.cx(0,1)
qtester.rz(0,[1])
qtester.cx(0,1)
qtester.h(0)
qtester.h(0)
qtester.h(1)
qtester.cx(0,1)
qtester.rz(0,[1])
qtester.cx(0,1)
qtester.h(0)
qtester.h(1)
qtester.qasm(filename = 'test_unittest')

hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}, hamiltonian_log_size=2).getOpenQASM(file_name='example_unittest', execution_time=0)

class TestFunction(unittest.TestCase):

    def test_pauliMatrix(self):
        self.assertTrue((hs.pauliMatrix('i') == np.eye(2)).all())
        self.assertFalse((hs.pauliMatrix('x') == np.array([[0,-1j],[1j,0]], dtype=complex)).all())

    def test_tensorProduct(self):
        self.assertTrue((hs.tensorProduct([np.eye(2),np.eye(2)]) == np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])).all())
        self.assertTrue((hs.tensorProduct([np.eye(2),np.array([[0,1],[1,0]])]) == np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])).all())
        
    def test_normalisation(self):
        self.assertTrue((hs.normalisation(np.array([1,0,0,0])) == np.array([1,0,0,0])).all())

    def test_trotterScalingModel(self):
        self.assertTrue(hs.trotterScalingModel(0) == 100)
        self.assertTrue(hs.trotterScalingModel(2, func_str='t**2') == 4)

    def test_NHS_getHamiltonian(self):
        self.assertTrue((hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'i':1}, hamiltonian_log_size=1).getHamiltonian() == np.eye(2)).all())
        self.assertTrue((hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xx':1}, hamiltonian_log_size=2).getHamiltonian() == np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])).all())

    def test_NHS_getExpHamiltonian(self):
        self.assertTrue((hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'i':1}, hamiltonian_log_size=1).getExpHamiltonian(execution_time=0) == np.eye(2)).all())
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}, hamiltonian_log_size=2).getExpHamiltonian(execution_time=np.pi),np.array([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]]),atol=1e-18).all())

    def test_NHS_getStatevector(self):
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}, hamiltonian_log_size=2).getStatevector(execution_time=np.pi, initial_state=np.array([1,0,0,0], dtype=complex)),np.array([-1,0,0,0], dtype=complex),atol=1e-18).all())
        # This test implies the statevector evolution test somewhat.
        # The test for ev. data will be hard and long winded plus i dont know the exact values so how can i be confident?

    def test_NHS_getEnergy(self):
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':2}, hamiltonian_log_size=2).getEnergy(execution_time=0, initial_state=np.array([1,0,0,0], dtype=complex)),1,atol=1e-18).all())
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}, hamiltonian_log_size=2).getEnergy(execution_time=np.pi, initial_state=np.array([1,0,0,0], dtype=complex)),1,atol=1e-18).all())

    def test_THS_getPartialQuantumCircuit(self):
        qc = QuantumCircuit(2)
        hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1}, hamiltonian_log_size=2).getPartialQuantumCircuit(quantum_circuit=qc, hamiltonian_term='xz', execution_time=0)

        qtester = QuantumCircuit(2)
        qtester.h(0)
        qtester.cx(0,1)
        qtester.rz(0,[1])
        qtester.cx(0,1)
        qtester.h(0)

        self.assertTrue(qc, qtester)

    def test_THS_getQuantumCircuit(self):
        qc = QuantumCircuit(2)
        hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}, hamiltonian_log_size=2).getQuantumCircuit(quantum_circuit=qc, trotter_step_count=1,execution_time=0)

        qtester = QuantumCircuit(2)
        qtester.h(0)
        qtester.cx(0,1)
        qtester.rz(0,[1])
        qtester.cx(0,1)
        qtester.h(0)
        qtester.h(0)
        qtester.h(1)
        qtester.cx(0,1)
        qtester.rz(0,[1])
        qtester.cx(0,1)
        qtester.h(0)
        qtester.h(1)

        self.assertTrue(qc, qtester)

    def test_THS_getOpenQASM(self):
        self.assertListEqual(list(open('/Users/gabewaite/Desktop/githubLateXclone/QSP-Coding/Python/example_unittest')),list(open('/Users/gabewaite/Desktop/githubLateXclone/QSP-Coding/Python/test_unittest')))
        # Idk how to do this...

    def test_THS_getStatevector(self):
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}, hamiltonian_log_size=2).getStatevector(execution_time=0, initial_state=np.array([1,0,0,0], dtype=complex)),np.array([1,0,0,0], dtype=complex),atol=1e-18).all())
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}, hamiltonian_log_size=2).getStatevector(execution_time=np.pi, initial_state=np.array([1,0,0,1], dtype=complex)),np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)], dtype=complex),atol=1e-18).all())
        # If these tests run True and the numerical tests run True then we can deduce that the inner product between then is also a correct quantity.
        # To this end, I am not going to generate a series of examples to test the statevectorEvolutionData file as we already know the components
        # of this function work...
        # It seems like a pointless endevour to check this. Moreover, i would have to analytically check these values are correct, which is not within my time frame unfortunately.

    def test_THS_getEnergy(self):
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}, hamiltonian_log_size=2).getEnergy(execution_time=np.pi, initial_state=np.array([1,0,0,0], dtype=complex)),0,atol=1e-18))
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':3}, hamiltonian_log_size=2).getEnergy(execution_time=1, initial_state=np.array([1,0,0,1], dtype=complex)),3,atol=1e-18).all())

if __name__ == '__main__':
    unittest.main()
    
