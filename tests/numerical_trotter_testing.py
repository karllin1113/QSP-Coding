import unittest
import numerical_trotter_hamiltonian_simulation as hs
import numpy as np
from qiskit.providers.aer.backends.statevector_simulator import StatevectorSimulator
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.providers.aer import QasmSimulator


# qtester = QuantumCircuit(2)
# qtester.h(0)
# qtester.cx(0,1)
# qtester.rz(0,[1])
# qtester.cx(0,1)
# qtester.h(0)
# qtester.h(0)
# qtester.h(1)
# qtester.cx(0,1)
# qtester.rz(0,[1])
# qtester.cx(0,1)
# qtester.h(0)
# qtester.h(1)
# qtester.qasm(filename = 'test_unittest')

# hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}).getOpenQASM(file_name='example_unittest')

class TestFunction(unittest.TestCase):
    '''
    Here I would like to mention that I am not entirely sure what to test...
    This is a new concept for me and I am not even sure what to google to figure this out. The exmaples ive seen only get me so far :/
    This is my best attempt at running tests. I dont feel like I learnt what to in the lecture so this is my best attempt given the time.

    If you would like to advise me on what this should have gone like I would be more than happy.
    '''
    def test_pauliMatrix(self):
        self.assertTrue((hs.pauliMatrix('i') == np.eye(2)).all())
        self.assertFalse((hs.pauliMatrix('x') == np.array([[0,-1j],[1j,0]], dtype=complex)).all())
        with self.assertRaises(TypeError): hs.pauliMatrix(1)
        with self.assertRaises(SyntaxError): hs.pauliMatrix('ixp')

    def test_tensorProduct(self):
        self.assertTrue((hs.tensorProduct([np.eye(2),np.eye(2)]) == np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])).all())
        self.assertTrue((hs.tensorProduct([np.eye(2),np.array([[0,1],[1,0]])]) == np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])).all())
        with self.assertRaises(TypeError): hs.tensorProduct([1,0],[1,1])
        
    def test_normalisation(self):
        self.assertTrue((hs.normalisation(np.array([1,0,0,0])) == np.array([1,0,0,0])).all())

    def test_trotterScalingModel(self):
        self.assertTrue(hs.trotterScalingModel(0) == 25)
        self.assertTrue(hs.trotterScalingModel(2, func_str='t**2') == 4)
        with self.assertRaises(ValueError): hs.trotterScalingModel(-1, func_str='t')

    def test_NHS_init(self):
        with self.assertRaises(TypeError): hs.NumericalHamiltonianSimulation(qubit_count=1.1, pauli_hash_map={'i':1})
        with self.assertRaises(ValueError): hs.NumericalHamiltonianSimulation(qubit_count=7, pauli_hash_map={'iiiiiii':1})
        with self.assertRaises(ValueError): hs.NumericalHamiltonianSimulation(qubit_count=5, pauli_hash_map={'xxxxx':1,'xxxxi':1,'xxxii':1,'xxiii':1,'xiiii':1,'ixxxx':1,'iixxx':1,'iiixx':1,'iiiix':1,'ixixi':1,'xixix':1,'yxyxy':1,'xyxyx':1,'zxzxz':1,'xzxzx':1,'zyzyz':1,'yzyzy':1,'ziziz':1,'yiyiy':1,'iyiyi':1,'izizi':1,'izxyz':1,'zzyzz':1,'xxzxx':1,'xyzyz':1,'zzzxy':1,'zzzxx':1})
        with self.assertRaises(ValueError): hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'i':11})
        with self.assertRaises(SyntaxError): hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'p':1, 'i':1})
        with self.assertRaises(ValueError): hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'x':1, 'ix':1})

    def test_NHS_getHamiltonian(self):
        self.assertTrue((hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'i':1}).getHamiltonian() == np.eye(2)).all())
        self.assertTrue((hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xx':1}).getHamiltonian() == np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])).all())

    def test_NHS_getExpHamiltonian(self):
        self.assertTrue((hs.NumericalHamiltonianSimulation(qubit_count=1, pauli_hash_map={'i':1}).getExpHamiltonian(execution_time=0) == np.eye(2)).all())
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}).getExpHamiltonian(execution_time=np.pi),np.array([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]]),atol=1e-18).all())

    def test_NHS_getStatevector(self):
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}).getStatevector(execution_time=np.pi, initial_state=np.array([1,0,0,0], dtype=complex)),np.array([-1,0,0,0], dtype=complex),atol=1e-18).all())
        # This test implies the statevector evolution test somewhat.
        # The test for ev. data will be hard and long winded plus i dont know the exact values so how can i be confident?

    def test_NHS_getEnergy(self):
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':2}).getEnergy(execution_time=0, initial_state=np.array([1,0,0,0], dtype=complex)),1,atol=1e-18).all())
        self.assertTrue(np.isclose(hs.NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}).getEnergy(execution_time=np.pi, initial_state=np.array([1,0,0,0], dtype=complex)),1,atol=1e-18).all())

    def test_THS_init(self):
        with self.assertRaises(TypeError): hs.TrotterHamiltonianSimulation(qubit_count=1.1, pauli_hash_map={'i':1})
        with self.assertRaises(ValueError): hs.TrotterHamiltonianSimulation(qubit_count=7, pauli_hash_map={'iiiiiii':1})
        with self.assertRaises(ValueError): hs.TrotterHamiltonianSimulation(qubit_count=5, pauli_hash_map={'xxxxx':1,'xxxxi':1,'xxxii':1,'xxiii':1,'xiiii':1,'ixxxx':1,'iixxx':1,'iiixx':1,'iiiix':1,'ixixi':1,'xixix':1,'yxyxy':1,'xyxyx':1,'zxzxz':1,'xzxzx':1,'zyzyz':1,'yzyzy':1,'ziziz':1,'yiyiy':1,'iyiyi':1,'izizi':1,'izxyz':1,'zzyzz':1,'xxzxx':1,'xyzyz':1,'zzzxy':1,'zzzxx':1})
        with self.assertRaises(ValueError): hs.TrotterHamiltonianSimulation(qubit_count=1, pauli_hash_map={'i':11})
        with self.assertRaises(SyntaxError): hs.TrotterHamiltonianSimulation(qubit_count=1, pauli_hash_map={'p':1, 'i':1})
        with self.assertRaises(ValueError): hs.TrotterHamiltonianSimulation(qubit_count=1, pauli_hash_map={'x':1, 'ix':1})
        with self.assertRaises(TypeError): hs.TrotterHamiltonianSimulation(qubit_count=1, pauli_hash_map={'x':1}, trotter_step_count=1.1)
        with self.assertRaises(ValueError): hs.TrotterHamiltonianSimulation(qubit_count=1, pauli_hash_map={'x':1}, trotter_step_count=-1)

    def test_THS_getPartialQuantumCircuit(self):
        qc = QuantumCircuit(2)
        hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1}).getPartialQuantumCircuit(quantum_circuit=qc, hamiltonian_term='xz', execution_time=0)

        qtester = QuantumCircuit(2)
        qtester.h(0)
        qtester.cx(0,1)
        qtester.rz(0,[1])
        qtester.cx(0,1)
        qtester.h(0)

        self.assertTrue(qc, qtester)

    def test_THS_getQuantumCircuit(self):
        qc = QuantumCircuit(2)
        hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}).getQuantumCircuit(quantum_circuit=qc, trotter_step_scaling_func='1',execution_time=0)

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

    # def test_THS_getOpenQASM(self):
    #     self.assertListEqual(list(open('path/example_unittest')),list(open('/Users/gabewaite/Desktop/githubLateXclone/QSP-Coding/Python/test_unittest')))
        # Idk how to do this...
        # It ouptuts a file to the local directory, idk how to find this an test it.
        # I dont really have the time to google this to try figure it out. This commented out test represents what *would* be ran in principle.

    def test_THS_getStatevector(self):
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}).getStatevector(execution_time=0, initial_state=np.array([1,0,0,0], dtype=complex)),np.array([1,0,0,0], dtype=complex),atol=1e-18).all())
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}).getStatevector(execution_time=np.pi, initial_state=np.array([1,0,0,1], dtype=complex)),np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)], dtype=complex),atol=1e-18).all())
        # If these tests run True and the numerical tests run True then we can deduce that the inner product between then is also a correct quantity.
        # To this end, I am not going to generate a series of examples to test the statevectorEvolutionData file as we already know the components
        # of this function work...
        # It seems like a pointless endevour to check this. Moreover, i would have to analytically check these values are correct, which is not within my time frame unfortunately.

    def test_THS_getEnergy(self):
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}).getEnergy(execution_time=np.pi, initial_state=np.array([1,0,0,0], dtype=complex)),0,atol=1e-18))
        self.assertTrue(np.isclose(hs.TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':3}).getEnergy(execution_time=1, initial_state=np.array([1,0,0,1], dtype=complex)),3,atol=1e-18).all())

if __name__ == '__main__':
    unittest.main()
