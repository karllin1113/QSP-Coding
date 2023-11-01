import numpy as np
import scipy as sp
import csv
import sympy

from sympy import Matrix
from sympy.physics.quantum.dagger import Dagger

import matplotlib.pyplot as plt

from qiskit.providers.aer.backends.statevector_simulator import StatevectorSimulator
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.providers.aer import QasmSimulator

"""
Here we outline some of the support information and functions. 

These functions help streamline repeated processes in the larger functions.

"""

pauli_terms_list = ['I', 'X', 'Y', 'Z']

def pauliMatrix(pauli_str: str) -> np.array:
    """
    Map a valid input string to the associated Pauli matrix.

    Args:
        pauli_str (str): A string input from the set {I, X, Y, Z}.

    Returns:
        matrix (np.array): A 2x2 np.array representing Pauli-'str'.

    Raises:
        TypeError: if the input is not a string.
        SyntaxError: if the terms if the input string are not from the allowed set.
    """
    if not isinstance(pauli_str, str):
        raise TypeError('The input must be a string.')
    
    pauli_str = pauli_str.upper()
    if any(part not in pauli_terms_list for part in pauli_str):
        raise SyntaxError('The term: {}, is an invalid Pauli operator!\nTry again using combinations of I, X, Y and Z.'.format(pauli_str))
    
    hash_map = {
        'I': np.eye(2),
        'X': np.array([[0,1],[1,0]], dtype=complex),
        'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
        'Z': np.array([[1,0],[0,-1]], dtype=complex)
        }

    return hash_map.get(pauli_str)

def tensorProduct(matrix_list: list) -> np.array:
    """
    Calculate the tensor product of a list of matrices from RIGHT to LEFT. Specifically,
        list = [A, B, C], the function computes:
        A ox B ox C

    Args:
        matrix_list (list): A list of matrices (np.array) to compute the tensor
            product.

    Returns:
        tensored matrix (np.array): A large tensor prodoct matrix.

    Raises:
        TypeError: if the input is not a list.

    Note:
        yu want the matrix for H o I, input [H, I]
    """
    if not isinstance(matrix_list, list):
        raise TypeError('The input must be a list!')
    
    product = matrix_list[-1]

    for matrix in matrix_list[-2::-1]:
        product = np.kron(matrix, product)

    return product

def normalisation(state):
    """
    Normalise an input vector using the L2 norm.

    Returns:
        normalised state: the input normalised by the L2 norm.
    """
    return state / np.linalg.norm(state)

def J() -> UnitaryGate:
    """
    Calculate the matrix J := S x Had.

    Returns:
        UnitaryGate : A QISKIT unitary gate of the matrix J.
    """
    matrix = np.array(
            [
                [1/np.sqrt(2),-1j/np.sqrt(2)],
                [1j/np.sqrt(2),-1/np.sqrt(2)]
            ],
            dtype = complex
        )
    return UnitaryGate(Operator(matrix), label = 'J')

def trotterScalingModel(time: float, func_varibale: str = 't',func_str: str = '25 + 0.2 * t ** 2'):
    """
    A function that evaluates an input function at a specified time.

    Args:
        time (float): the input time for function evaluation.
        func_variable (str): the variable used in the input function.
        func_str (str): the string representing the function in sympy format. This function must
            evalue to a positive value for all input values > 0.

    Returns: 
        integer value (int): the evaluated function at given time rounded to the nearest integer.

    Note:
        It is assumed the function will evaluate to a positive value.

    Caution:
        The number of Trotter steps is uncapped by this function so proceed with caution. A larger
            number of steps equates to a long running time.
    """
    t = sympy.symbols(func_varibale)

    input_func = sympy.sympify(func_str)
    result = input_func.subs(t, time)

    step_scaling = result.evalf()

    if step_scaling < 0:
        raise ValueError('Your input function must evaluate to a positive value for all time > 0.')

    return int(step_scaling)

class NumericalHamiltonianSimulation:
    """
    A class for calculating the 'classical' (numerical) evolution of a Hamiltonian.
    
    Args:
        qubit_count (int): The number of qubits the input Hamiltonian acts on. Maximum
          value is 5

        > For example:

            If Hamiltonian is 2**3 x 2**3, then 3 qubits are needed

        pauli_hasp_map (dict): A dictionary specifying the Pauli terms and their coefficients.
            The Pauli string terms are automatrically capitalised. Number of terms in the dictionary
            is bounded by 20. The coefficients must be REAL numbers in [-5,5]

        > For example:

            If H = 2*ZZI + 3*XYY + 0.69*XIZ + ..., the associated pauli_hash_map would be,
                {'ZZI':2, 'XYY':3, 'XIZ:0.69, ...}.

    Raises:
        TypeError: if the qubit_count is not an integer.
        ValueError: if qubit_count exceeds 5.
        ValueError: if if any of the characters in the pauli_hash_map keys are not 
          in pauli_terms_list = ['I', 'X', 'Y', 'Z'].
        ValueError: if any of the pauli_hash_map keys have a length exceeding qubit_count.
        SyntaxError: if any characters in the pauli_hash_map key strings are not in the allowed set.

    Example:
        (?) Get the Hamiltonian for an input Pauli_hash_map:

        > NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}).getHamiltonian()

        >>> array([[ 0.+0.j,  0.+0.j,  1.+0.j,  1.+0.j],
                    [ 0.+0.j,  0.+0.j,  1.+0.j, -1.+0.j],
                    [ 1.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                    [ 1.+0.j, -1.+0.j,  0.+0.j,  0.+0.j]])

        #--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#

        (?) Get the statevector at time t=pi for some random initial state given some input pauli_hash_map:

        > NumericalHamiltonianSimulation(qubit_count=2, pauli_hash_map={'xz':1, 'xx':1}).getStatevector(execution_time=np.pi, initial_state=np.array([1,1j,-1j,1+1j], dtype=complex))

        >>> array([-0.11907301+0.30481274j,  0.60962549-0.42388575j,
                    -0.30481274+0.42388575j,  0.18573974+0.18573974j])
    """
    def __init__(
        self,
        qubit_count: int,
        pauli_hash_map: dict,
    ) -> None:

        self.qubit_count = qubit_count

        # Error check the qubit_count is an integer
        if not isinstance(qubit_count, int):
            raise TypeError('The qubit count must be an integer! Try again.')

        # Error check the qubit_count value
        if qubit_count > 6:
            raise ValueError('The number of qubits cannot exceed 5!\nPlease try again!')

        # Capitalise each string
        self.pauli_hash_map = {key.upper(): value for key, value in pauli_hash_map.items()}

        # Error check dictionary length
        if len(self.pauli_hash_map) > 20:
            raise ValueError('The max number of terms for the input dictionary is 20.\nTry again!')

        # Error check the coefficeint size
        if not all(-10 <= coefficient <= 10 for coefficient in self.pauli_hash_map.values()):
            raise ValueError('The coefficients must not exceed the limits, [-10,10]!\nTry again.')

        # Error check illegal terms in the strings
        for pauli_str in self.pauli_hash_map.keys():
            if any(part not in pauli_terms_list for part in pauli_str):
                 raise SyntaxError('The character: {}, is an invalid Pauli operator!\nTry again using combinations of I, X, Y and Z.'.format(pauli_str))
            elif len(pauli_str) > qubit_count:
                raise ValueError('The term: {} is an invalid Pauli operator!\nTry again making sure the number of terms does not exceed {}.'.format(pauli_str, qubit_count))

    def getHamiltonian(
        self
    ) -> np.array:
        """
        A function for constructing the matrix given the input dictionary.

        Returns: 
            Matrix (np.array): The Hamiltonian.

        Note: 
            Call sympy.Matrix() for fancy output
        """

        H = np.zeros([2 ** self.qubit_count, 2 ** self.qubit_count], dtype=complex)

        for pauli_str, coefficient in self.pauli_hash_map.items():
            term = list(map(pauliMatrix,pauli_str))
            tensor_prod_term = tensorProduct(term)
            H += coefficient * tensor_prod_term

        return H

    def getExpHamiltonian(
        self,
        execution_time: float = 2 * np.pi
    ):
        """
        A function for explicity calculating the exponential of the input matrix. 
          Calls from getHamiltonian().

        Args:
            execution_time (float): the evaluation time for the exponentiated Hamiltonian.

        Returns:
            Matrix (np.array): the exponentiated Hamiltonian at time t, e**(-i * t * H)

        Note: 
            Call sympy.Matrix() for fancy output
        """
        exp_H = sp.linalg.expm(-1j * execution_time * self.getHamiltonian())

        return exp_H

    def getStatevector(
            self,
            execution_time: float,
            initial_state: np.array,
        ) -> np.array:
        """
        A function for extracting the statevector at a given time. Calls 
          getExpHamiltonian().

        Args: 
            execution_time (float): the evaluation time for the statevector.
            initial_state (np.array): the initial statevector.
            decimal_points (int): the rounding value of decimal places.

        Returns: 
            evolved_state (np.array): a rounded statevector evolved via the exp. Hamiltonian.

        Note: 
            Call sympy.Matrix() for fancy output
        """
        # Normalise the inital_state
        initial_state = normalisation(initial_state)

        # Define the evolved state, v(t) = exp(-iHt)v(0)
        evolved_state = self.getExpHamiltonian(execution_time=execution_time).dot(initial_state)
        # Normalise the evolved state
        evolved_state = normalisation(evolved_state)
        # Return the rounded evolved state
        return normalisation(evolved_state)

    def statevectorEvolutionData(
        self,
        file_name: str,
        initial_state: np.array,
        max_time: int = 2 * np.pi,
        print_messages: bool = True
    ):
        """
        A function for saving statevector evolution data.

        Args: 
            file_name (str): name of the file.
            initial_state (np.array): the initial statevector.
            max_time (float): maximum time for the evolution.
            print_messages (bool): prints the message log.

        Returns: 
            files: a series of files for the statevector evolution data.
                1. metadata file (..._metadata.txt): a .txt file of the input data.
                2. data file (..._sv.csv): a .csv file of the plot data.
        """
        if max_time < 0:
            raise ValueError('The maximum time must exceed 0 seconds! Try again.')

            # check init state and ham are same size
        # Normalise the inital_state
        initial_state = normalisation(initial_state)

        # Define exp(-iHt) at time t
        exp_H = lambda t: self.getExpHamiltonian(execution_time=t)
        # Use exp_H(t) to define eveolved state at time t
        evolved_state = lambda t: exp_H(t).dot(initial_state)
        # Define the inner product at time t against the initial state
        inner_product = lambda t: np.vdot(initial_state, normalisation(evolved_state(t)))

        time_data = np.linspace(0, max_time, 100)

        if print_messages is True: print('>>> fetching probaility data...')
        probability_data = [abs(inner_product(t)) ** 2 for t in time_data]
        if print_messages is True: 
            print('>>> probaility data found!')
            print('>>> fetching final statevector and generating data!')

        data = [(a, x, y) for a, (x, y) in enumerate(zip(time_data, probability_data))]
        metadata = [
            'Simulation Type: {}'.format('Numerical'),
            'Hamiltonian Map: {}'.format(self.pauli_hash_map),
            'Qubit Count: {}'.format(self.qubit_count),
            'Execution Time: 0 to {}'.format(max_time),
            'Initial State: {}'.format(initial_state),
            'Final State: {}'.format(normalisation(evolved_state(max_time)))
        ]
        file_data = "numerical_{}_sv.csv".format(file_name)
        file_metadata = "numerical_{}_metadata.txt".format(file_name)

        with open(file_data, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time", "probability"])
            writer.writerows(data)

        with open(file_metadata, mode='w') as f:
          for item in metadata:
            f.write(item + '\n')

        print("Data saved to {}.\nMetadata saved to {}.".format(file_data, file_metadata))

    def getEnergy(
        self,
        execution_time: float,
        initial_state: np.array,
        decimal_points: int = 5
    ) -> float:
        """
        A function for extracting the energy at a given time. Calls 
          getExpHamiltonian() and getStatevector().

        Args: 
            execution_time (float): the evaluation time for the energy.
            initial_state (np.array): the initial statevector.
            decimal_points (int): the rounding value of decimal places.

        Returns: 
            evolved_obs (float): a rounded value for the energy observable.

        Note: 
            np.real() is used to remove any lingering imaginary values
        """
        # Normalise the inital_state
        initial_state = normalisation(initial_state)

        evolved_state = self.getStatevector(self.getExpHamiltonian(execution_time=execution_time), initial_state)
        evolved_state = normalisation(evolved_state)

        energy_obs = np.vdot(evolved_state, self.getExpHamiltonian().dot(evolved_state))

        # Remove imaginary part
        energy_obs = np.real(energy_obs)

        # Round it
        energy_obs = np.round(energy_obs, decimal_points)

        return energy_obs

    def energyEvolutionData(
        self,
        file_name: str,
        initial_state: np.array,
        max_time: int = 2 * np.pi,
        print_messages: bool = True
    ):
        """
        A function for saving energy evolution data.

        Args: 
            file_name (str): name of the file.
            initial_state (np.array): the initial statevector.
            max_time (float): maximum time for the evolution.
            print_messages (bool): prints the message log.

        Returns: 
            files: a series of files for the energy evolution data.
                1. metadata file (..._metadata.txt): a .txt file of the input data and energy values.
                2. data file (..._sv.csv): a .csv file of the plot data.
        """
        if max_time < 0:
            raise ValueError('The maximum time must exceed 0 seconds! Try again.')

        # check init state and ham are same size

        initial_state = normalisation(initial_state)

        exp_H = lambda  t: self.getExpHamiltonian(execution_time=t)

        evolved_state = lambda t: normalisation(exp_H(t).dot(initial_state)) # Ensure normalisation.

        energy_observable = lambda t: np.vdot(evolved_state(t), self.getHamiltonian().dot(evolved_state(t))) # <H>

        time_data = np.linspace(0, max_time, 100)

        if print_messages is True: print('>>> fetching energy data...')
        energy_data = [np.real(energy_observable(t)) for t in time_data]
        if print_messages is True: 
            print('>>> energy data found!')
            print('>>> fetching the final state and generating data!')

        data = [(a, x, y) for a, (x, y) in enumerate(zip(time_data, energy_data))]
        metadata = [
            'Simulation Type: {}'.format('Numerical'),
            'Hamiltonian Map: {}'.format(self.pauli_hash_map),
            'Qubit Count: {}'.format(self.qubit_count),
            'Execution Time: 0 to {}'.format(max_time),
            'Initial State: {}'.format(initial_state),
            'Initial Energy Value: {}'.format(energy_data[0]),
            'Final Energy Value: {}'.format(energy_data[-1]) 
        ]
        file_data = "numerical_{}_energy.csv".format(file_name)
        file_metadata = "numerical_{}_metadata.txt".format(file_name)

        with open(file_data, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time", "energy"])
            writer.writerows(data)

        with open(file_metadata, mode='w') as f:
          for item in metadata:
            f.write(item + '\n')

        print("Data saved to {}.\nMetadata saved to {}.".format(file_data, file_metadata))

class TrotterHamiltonianSimulation:
    """
    A class for calculating the Trotterisation of a Hamiltonian.
    
    Args:
        qubit_count (int): The number of qubits the input Hamiltonian acts on.

        > For example:

            If Hamiltonian is 2**3 x 2**3, then 3 qubits are needed

        pauli_hasp_map (dict): A dictionary specifying the Pauli terms and their coefficients.
            The Pauli string terms are automatrically capitalised.

        > For example:

            If H = 2*ZZI + 3*XYY + 6.9*XIZ + ..., the associated pauli_hash_map would be,
                {'ZZI':2, 'XYY':3, 'XIZ:6.9, ...}.

        trotter_step_coun (int): The number of Trotter steps used in the Trotter decomposition procedure.
            The default value is 25 and is expected to lie in the interval [0,500].

        > Note: if the Trotter scaling function is employed, the step count goes uncapped and is determined
            by user input. See trotterSclaingModel for more information.

    Raises:
        TypeError: if the qubit_count is not an integer.
        ValueError: if qubit_count exceeds 5.
        ValueError: if if any of the characters in the pauli_hash_map keys are not 
          in pauli_terms_list = ['I', 'X', 'Y', 'Z'].
        ValueError: if any of the pauli_hash_map keys have a length exceeding qubit_count.
        SyntaxError: if any characters in the pauli_hash_map key strings are not in the allowed set.
        TypeError: if the trotter_Step_count is not an integer.
        ValueError: if the trotter_step_count is outside the range [0,500]

    Example:
        (?) Get the statevector at time t=pi for some random initial state given some input pauli_hash_map:

        > TrotterHamiltonianSimulation(qubit_count=2, pauli_hash_map={'zz':1}).getStatevector(execution_time=np.pi, initial_state=np.array([1,0,0,1], dtype=complex))

        >>> Statevector([0.70710678+0.j, 0.        +0.j, 0.        +0.j,
                            0.70710678+0.j],
                            dims=(2, 2))
    
    """
    def __init__(
        self,
        qubit_count: int,
        pauli_hash_map: dict,
        trotter_step_count: int = 25
    ) -> None:

        self.qubit_count = qubit_count

        # Error check the qubit_count is an integer
        if not isinstance(qubit_count, int):
            raise TypeError('The qubit count must be an integer! Try again.')

        # Error check the qubit_count value
        if qubit_count > 6:
            raise ValueError('The number of qubits cannot exceed 5!\nPlease try again!')

        # Capitalise each string
        self.pauli_hash_map = {key.upper(): value for key, value in pauli_hash_map.items()}

        # Error check dictionary length
        if len(self.pauli_hash_map) > 20:
            raise ValueError('The max number of terms for the input dictionary is 20.\nTry again!')

        # Error check the coefficeint size
        if not all(-10 <= coefficient <= 10 for coefficient in self.pauli_hash_map.values()):
            raise ValueError('The coefficients must not exceed the limits, [-10,10]!\nTry again.')

        # Error check illegal terms in the strings
        for pauli_str in self.pauli_hash_map.keys():
            if any(part not in pauli_terms_list for part in pauli_str):
                 raise SyntaxError('The term: {}, is an invalid Pauli operator!\nTry again using combinations of I, X, Y and Z.'.format(pauli_str))
            elif len(pauli_str) > qubit_count:
                raise ValueError('The term: {} is an invalid Pauli operator!\nTry again making sure the number of terms does not exceed {}.'.format(pauli_str, qubit_count))

        self.trotter_step_count = trotter_step_count
        # Error check the Trotter step count
        if not isinstance(trotter_step_count, int):
            raise TypeError('The number of Trotter steps should be an integer!')

        if trotter_step_count <= 0 or trotter_step_count > 500:
            raise ValueError('The Trotter step count must lie between 0 and 500 inclusive!')

    def getHamiltonian(
        self
    ) -> np.array:
        """
        A function for constructing the matrix given the input dictionary.

        Returns: 
            Matrix (np.array): The Hamiltonian.

        Note: 
            Call sympy.Matrix() for fancy output
            This is NOT a quantum feature rather an numerical calculation.
        """

        H = np.zeros([2 ** self.qubit_count, 2 ** self.qubit_count], dtype=complex)

        for pauli_str, coefficient in self.pauli_hash_map.items():
            term = list(map(pauliMatrix,pauli_str))
            tensor_prod_term = tensorProduct(term)
            H += coefficient * tensor_prod_term

        return H

    def getPartialQuantumCircuit(
        self,
        quantum_circuit: QuantumCircuit,
        hamiltonian_term: str,
        execution_time: float = 2 * np.pi
    ) -> QuantumCircuit:
        """
        A function to construct the exponential quantum circuit of a valid Pauli string at a specified time.

        Args:
            quantum_circuit (QuantumCircuit): the quantum circuit to run the function.
            hamiltonian_term (str): the valid Hamiltonian term (from the input dictionary) for which the circuit is constructed.
            execution_time (float): the evaluation time for the circuit.

        Returns:
            quantum circuit (QuantumCircuit): a quantum circuit for the exponentiation of the input Pauli term at the given time.
        """

        hamiltonian_term = hamiltonian_term.upper()
        # Check key in provided dictionary
        # This handles the size and the correct terms required
        if hamiltonian_term not in self.pauli_hash_map:
            raise KeyError("The key {} is not in the input dictionary!\nPick an valid option".format(hamiltonian_term))

        # Create list of positions for non-identity gates
        non_identity_positions = []

        # Check which positions are not the identity and
        # add them to non_identity_positions
        for index, char in enumerate(hamiltonian_term):
            if char != 'I':
                non_identity_positions.append(index)

        for register, pauli_term in enumerate(hamiltonian_term):
            if pauli_term == 'X':
                quantum_circuit.h(register)
            elif pauli_term == 'Y':
                quantum_circuit.append(J(), [register])

        # The CXGate must have target on the last entry of non_identity_positions
        # since this marks the final non identity Pauli which takes the RZGate
        for index in range(len(non_identity_positions) - 1):
            quantum_circuit.cx(non_identity_positions[index], non_identity_positions[index+1])

        # RZGate(t) = e**(-i t/2 z) on non_identity_positions[-1]
        # If non_identity_positions empty, do Identity gate
        if len(non_identity_positions) != 0:
            quantum_circuit.rz(2 * execution_time, [non_identity_positions[-1]])

        # Reverse the CXGates
        for index in range(len(non_identity_positions)-1,0,-1):
            quantum_circuit.cx(non_identity_positions[index - 1], non_identity_positions[index])

        for register, pauli_term in enumerate(hamiltonian_term):
            if pauli_term == 'X':
                quantum_circuit.h(register)
            elif pauli_term == 'Y':
                quantum_circuit.append(J(), [register])

    def getQuantumCircuit(
        self,
        quantum_circuit: QuantumCircuit,
        trotter_step_scaling_func: 'sympy_func' = None,
        execution_time: float = 2 * np.pi,
    ) -> QuantumCircuit:

        """
        A function for constructing the full quantum circuit for the input Hamiltonian.

        Args:
            quantum_circuit (QuantumCircuit): the quantum circuit to run the function.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            execution_time (float): the evaluation time for the circuit.
        
        Returns:
            quantum circuit (QuantumCircuit): a circuit for the exponentiation of the full input Hamiltonian decomposed via
              the Trotterisation method.

        Note:
            The trotter_step_scaling_func is designed to be used to achive a better approximation. It can be used to increase the 
              number of Trotter steps as time evolves or used alternatively to specify a constant number of steps.
        """
        if trotter_step_scaling_func is not None:
            trotter_steps = trotterScalingModel(execution_time, func_str=trotter_step_scaling_func)
        else:
            trotter_steps = self.trotter_step_count

        # Define a circuit to append the partial circuits to
        trotter_circuit = QuantumCircuit(self.qubit_count)

        # Iterate throught the pauli_hash_map to define the quantum circuits
        # for each of the hamiltonian terms and their respective coefficients
        for hamiltonian_term, coefficient in self.pauli_hash_map.items():
            rescaled_time = coefficient * execution_time / trotter_steps # Rescale time for the Trotter decomp.
            self.getPartialQuantumCircuit(quantum_circuit=trotter_circuit,\
                                          hamiltonian_term = hamiltonian_term,\
                                          execution_time = rescaled_time)
        # Join each Trotter step circuit together [trotter_steps] times
        for count in range(trotter_steps):
            quantum_circuit.compose(trotter_circuit, [i for i in range(self.qubit_count)], inplace=True)

    def getOpenQASM(
            self,
            file_name: str
    ) -> "file":
        """
        A function for saving the OpenQASM code for the Hamiltonian QuantumCircuit.

        Args:
            file_name (str): the name of the file.
            execution_time (float): the evaluation time for the quantum circuit.

        Returns:
            A file with the OpenQASM (QIKSIT) code of the quantum circuit for the input Hamiltonian at a given 
              execution time with only a single Trotter step.

        Note: 
            Only a single Trotter step is given as this can be repeated for personal simulation.
        """

        qc = QuantumCircuit(self.qubit_count)
        self.getQuantumCircuit(quantum_circuit=qc, trotter_step_scaling_func='t', execution_time=1)
        output_file_name = 'trotter_qasm_{}.txt'.format(file_name)
        qc.qasm(filename = output_file_name)

        return "File saved as {}!".format(file_name)

    def getTrotterSimulation(
        self,
        quantum_circuit: QuantumCircuit,
        trotter_step_count: int = None,
        trotter_step_scaling_func: 'sympy_func' = None,
        execution_time: float = 2 * np.pi,
        initial_state: np.array = None,
        print_messages: bool = True
    ) -> None:
        """
        A function for obtaining the simulation of the Trotter decomposition of the input Hamiltonian.

        Args:
            quantum_circuit (QuantumCircuit): the quantum circuit to run the function.
            trotter_step_count (int): the number of Trotter steps.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            execution_time (float): the evaluation time for the circuit.
            initial_state (np.array): the initial state of the system.
            print_messages (bool): prints the message log.

        Returns:
            The QISKIT simulation data for the quantum circuit.

        Caution: 
            A large execution_time will result in longer running times. Proceed with caution.
        """
        if trotter_step_count is None: trotter_step_count = self.trotter_step_count

        if trotter_step_scaling_func is not None:
            trotter_step_count = trotterScalingModel(execution_time, trotter_step_scaling_func)

        # Check intial_state is correct size
        if int(np.log2(len(initial_state))) != self.qubit_count:
            raise ValueError('The size of the initial state must be 2 ** {}'.format(self.qubit_count))

        if print_messages is True:
            print('###-- Simulation Initialised --###\n')
            print('> Trotter step count: {}'.format(trotter_step_count))
            print('> Execution time: {}'.format(execution_time))
            print('> Initial state: {}\n'.format(initial_state))
            print('>>> building circuit...')
            print('>>> initialising state...')

        # Normalise the inital_state
        initial_state = normalisation(initial_state)
        if print_messages is True: print('>>> normalised initial state: {}'.format(initial_state))

        # Generate the inital_state based on circuit input
        quantum_circuit.initialize(initial_state, [i for i in range(self.qubit_count)])

        self.getQuantumCircuit(quantum_circuit=quantum_circuit, trotter_step_count=trotter_step_count, execution_time=execution_time)

        quantum_circuit = quantum_circuit.reverse_bits()

        if print_messages is True:
          print('>>> circuit built!')
          print('>>> fetching simultaion...')

        backend = Aer.get_backend('statevector_simulator')
        result  = execute(quantum_circuit, backend).result()

        if print_messages is True: print('###-- Simulation Complete! --###')

        return result

    def getStatevector(
        self,
        execution_time: float,
        initial_state: np.array,
        trotter_step_scaling_func: 'sympy_func' = None
    ) -> np.array:
        """
        A function for extracting the statevector at a given time. Calls 
          getQuantumCircuit().

        Args: 
            execution_time (float): the evaluation time for the statevector.
            initial_state (np.array): the initial statevector.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            decimal_points (int): the rounding value of decimal places.

        Returns: 
            evolved_state (np.array): a rounded statevector evolved via the simulation.

        Note: 
            Call sympy.Matrix() for fancy output

        Caution: 
            A large execution_time will result in longer running times. Proceed with caution.
        """
       
        # Normalise intial_statevector
        initial_state = normalisation(initial_state)
        # Define quantum circuit
        qc = QuantumCircuit(self.qubit_count)
        # Generate the inital_state based on circuit input
        qc.initialize(initial_state, [i for i in range(self.qubit_count)])
        qc = qc.reverse_bits()
        self.getQuantumCircuit(quantum_circuit=qc, execution_time=execution_time, trotter_step_scaling_func = trotter_step_scaling_func)
        qc = qc.reverse_bits()

        backend = Aer.get_backend('statevector_simulator')
        result  = execute(qc, backend).result()

        statevector = result.get_statevector(qc)
        statevector = normalisation(statevector)

        return statevector

    def statevectorEvolutionData(
        self,
        file_name: str,
        initial_state: np.array,
        trotter_step_scaling_func: 'sympy_func' = None,
        max_time: int = 2 * np.pi,
        print_messages: bool = True,
        linspace_count: int = 100
    ):
        """
        A function for saving statevector evolution data. Calls getStatevector().

        Args: 
            file_name (str): name of the file.
            initial_state (np.array): the initial statevector.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            max_time (float): maximum time for the evolution.
            print_messages (bool): prints the message log.
            linspace_count (int): the number of steps in the linspace array.

        Returns: 
            files: a series of files for the statevector evolution data.
                1. metadata file (..._metadata.txt): a .txt file of the input data.
                2. data file (..._sv.csv): a .csv file of the plot data.

        Caution: 
            A large max_time with fixed linspace_count will result in sparse data. Proceed with caution.
            A large number of linspace_counts will result in a longer running time. Proceed with caution.
        """

        if max_time < 0:
            raise ValueError('The maximum time must exceed 0 seconds! Try again.')

            # check init state and ham are same size

        # Normalise the inital_state
        initial_state = normalisation(initial_state)

        evolved_state = lambda t: self.getStatevector(execution_time=t, initial_state=initial_state, trotter_step_scaling_func=trotter_step_scaling_func) # Already normalised.

        inner_product = lambda t: np.vdot(initial_state, evolved_state(t))

        time_data = np.linspace(0, max_time, linspace_count)

        if print_messages is True: print('>>> fetching probaility data...')
        probability_data = [abs(inner_product(t)) ** 2 for t in time_data]
        if print_messages is True: 
            print('>>> probability data found!')
            print('>>> fetching final state and generating data...')

        data = [(a, x, y) for a, (x, y) in enumerate(zip(time_data, probability_data))]
        metadata = [
            'Simulation Type: {}'.format('Trotter'),
            'Hamiltonian Map: {}'.format(self.pauli_hash_map),
            'Qubit Count: {}'.format(self.qubit_count),
            'Execution Time: 0 to {}'.format(max_time),
            'Number of Trotter Steps: {}'.format(self.trotter_step_count),
            'Trotter Scaling Function: {}'.format(trotter_step_scaling_func),
            'Initial State: {}'.format(initial_state),
            'Final State: {}'.format(normalisation(evolved_state(max_time)))
        ]
        file_data = "trotter_{}_sv.csv".format(file_name)
        file_metadata = "trotter_{}_metadata.txt".format(file_name)

        with open(file_data, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time", "probability"])
            writer.writerows(data)

        with open(file_metadata, mode='w') as f:
          for item in metadata:
            f.write(item + '\n')

        print("Data saved to {}.\nMetadata saved to {}.".format(file_data, file_metadata))

    def getEnergy(
        self,
        execution_time: float,
        initial_state: np.array,
        trotter_step_scaling_func: 'sympy_func' = None,
        decimal_points: int = 5
    ) -> float:
        """
        A function for extracting the energy at a given time. 

        Args: 
            file_name (str): name of the file.
            initial_state (np.array): the initial statevector.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            max_time (float): maximum time for the evolution.
            print_messages (bool): prints the message log.

        Returns: 
            evolved_obs (float): a rounded value for the energy observable.

        Note: 
            np.real() is used to remove any lingering imaginary values.
            It is not possible to perform the Hadamard test in general as H is not a unitary.

        Caution: 
            A large execution_time will result in longer running times. Proceed with caution.
        """
        # Normalise the inital_state
        initial_state = normalisation(initial_state)

        evolved_state = self.getStatevector(execution_time=execution_time, initial_state=initial_state, trotter_step_scaling_func=trotter_step_scaling_func) # Already normalised.

        energy_obs = np.dot(np.conjugate(evolved_state), self.getHamiltonian().dot(evolved_state))

        # Remove imaginary part
        energy_obs = np.real(energy_obs)

        # Round it
        energy_obs = np.round(energy_obs, decimal_points)

        return energy_obs

    def energyEvolutionData(
        self,
        file_name: str,
        initial_state: np.array,
        trotter_step_scaling_func: 'sympy_func' = None,
        max_time: int = 2 * np.pi,
        print_messages: bool = True,
        linspace_count: int = 100
    ):
        """
        A function for saving energy evolution data.

        Args: 
            file_name (str): name of the file.
            initial_state (np.array): the initial statevector.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            max_time (float): maximum time for the evolution.
            print_messages (bool): prints the message log.

        Returns: 
            files: a series of files for the energy evolution data.
                1. metadata file (..._metadata.txt): a .txt file of the input data.
                2. data file (..._sv.csv): a .csv file of the plot data.

        Note:
            It is not possible to perform the Hadamard test in general as H is not a unitary.

        Caution: 
            A large max_time with fixed linspace_count will result in sparse data. Proceed with caution.
            A large number of linspace_counts will result in a longer running time. Proceed with caution.
        """

        if max_time < 0:
            raise ValueError('The maximum time must exceed 0 seconds! Try again.')

        # check init state and ham are same size

        H = self.getHamiltonian()

        evolved_state = lambda t: self.getStatevector(execution_time=t, initial_state=initial_state, trotter_step_scaling_func=trotter_step_scaling_func) # Already normalised.

        H_x_evovled_state = lambda t: H.dot(evolved_state(t))

        energy_observable = lambda t: np.vdot(evolved_state(t), H_x_evovled_state(t))

        time_data = np.linspace(0, max_time, linspace_count)

        if print_messages is True: print('>>> fetching energy data...')
        # Call np.real to remove any residual complex error
        energy_data = [np.real(energy_observable(t)) for t in time_data]
        if print_messages is True: 
            print('>>> energy data found!')
            print('>>> fetching final energy and generating data...')

        data = [(a, x, y) for a, (x, y) in enumerate(zip(time_data, energy_data))]
        metadata = [
            'Simulation Type: {}'.format('Trotter'),
            'Hamiltonian Map: {}'.format(self.pauli_hash_map),
            'Qubit Count: {}'.format(self.qubit_count),
            'Execution Time: 0 to {}'.format(max_time),
            'Number of Trotter Steps: {}'.format(self.trotter_step_count),
            'Trotter Scaling Function: {}'.format(trotter_step_scaling_func),
            'Initial State: {}'.format(initial_state),
            'Initial Energy Value: {}'.format(energy_data[0]),
            'Final Energy Value: {}'.format(energy_data[-1])
        ]
        file_data = "trotter_{}_energy.csv".format(file_name)
        file_metadata = "trotter_{}_metadata.txt".format(file_name)

        with open(file_data, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time", "energy"])
            writer.writerows(data)

        with open(file_metadata, mode='w') as f:
          for item in metadata:
            f.write(item + '\n')

        print("Data saved to {}.\nMetadata saved to {}.".format(file_data, file_metadata))

    def NTFidelityTest(
        self,
        file_name: str,
        initial_state: np.array,
        trotter_step_scaling_func: 'sympy_func' = None,
        max_time: int = 2 * np.pi,
        print_messages: bool = True,
        linspace_count: int = 100
    ): 
        """
        A function for saving numerical and Trotter statevector fidelity.

        Args: 
            file_name (str): name of the file.
            initial_state (np.array): the initial statevector.
            trotter_step_scaling_func (sympy_func): a Trotter step sclaing function with time as a variable.
            max_time (float): maximum time for the evolution.
            print_messages (bool): prints the message log.

        Returns: 
            files: a series of files for the fidelity data.
                1. metadata file (..._metadata.txt): a .txt file of the input data.
                2. data file (..._fidelity.csv): a .csv file of the plot data.

        Note:
            The numerical calculation is automatically called in this function.

        Caution: 
            A large max_time with fixed linspace_count will result in sparse data. Proceed with caution.
            A large number of linspace_counts will result in a longer running time. Proceed with caution.
        """

        if max_time < 0:
            raise ValueError('The maximum time must exceed 0 seconds! Try again.')
        
        if print_messages is True: print('>>> initialising the numerical statevector...')
        numerical_steup = NumericalHamiltonianSimulation(qubit_count=self.qubit_count, pauli_hash_map=self.pauli_hash_map)

        numerical_statevector = lambda t: numerical_steup.getStatevector(execution_time=t, initial_state=initial_state)

        trotter_statevector = lambda t: self.getStatevector(execution_time=t, initial_state=initial_state, trotter_step_scaling_func=trotter_step_scaling_func)

        fidelity = lambda t: abs(np.vdot(numerical_statevector(t), trotter_statevector(t))) ** 2

        time_data = np.linspace(0, max_time, linspace_count)

        if print_messages is True: print('>>> fetching fidelity data...')
        fidelity_data = [fidelity(t) for t in time_data]
        if print_messages is True: 
            print('>>> fidelity data found!')
            print('>>> fetching final statevectors and generating data...')

        data = [(a, x, y) for a, (x, y) in enumerate(zip(time_data, fidelity_data))]
        metadata = [
            'Simulation Type: {}'.format('Trotter vs Numerical'),
            'Hamiltonian Map: {}'.format(self.pauli_hash_map),
            'Qubit Count: {}'.format(self.qubit_count),
            'Execution Time: 0 to {}'.format(max_time),
            'Number of Trotter Steps: {}'.format(self.trotter_step_count),
            'Trotter Scaling Function: {}'.format(trotter_step_scaling_func),
            'Initial State: {}'.format(initial_state),
            'Final Numerical Statevector: {}'.format(normalisation(numerical_statevector(max_time))),
            'Final Trotter Statevector: {}'.format(normalisation(trotter_statevector(max_time)))
        ]
        file_data = "{}_NT_fidelity.csv".format(file_name)
        file_metadata = "{}_metadata.txt".format(file_name)

        with open(file_data, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "time", "fidelity"])
            writer.writerows(data)

        with open(file_metadata, mode='w') as f:
          for item in metadata:
            f.write(item + '\n')

        print("Data saved to {}.\nMetadata saved to {}.".format(file_data, file_metadata))
