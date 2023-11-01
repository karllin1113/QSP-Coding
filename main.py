from modules import numerical_trotter_hamiltonian_simulation as hs
from modules import runHamiltonianSimulatorbyQET as ks


import numpy as np
import scipy as sp
import csv
import sympy

from sympy import Matrix
from sympy.physics.quantum.dagger import Dagger

import matplotlib.pyplot as plt

initial_state = None
execution_time = None
qubit_number = None

error_global = None
unitary_matrix = None
product = None

hamitlonian = None
normalized_hamiltonian = None

hash_map = None
U_block = None



zero = np.array([1,0])
one = np.array([0,1])
V = np.kron(zero,zero)
qubit = 4
binary_state = qubit*[]

def other_inputs():
    global initial_state
    global execution_time
    global qubit_number
    state = np.array(eval(input("Enter the initial state in [...] format: ")))
    initial_state = state
    time = float(input("Enter the time between 0 and 5 to execute "))
    if time>5 and time<0:
        raise ValueError("Please choose the execution time between 0 and 5 ")
    execution_time = time
    qubit = int(input("Enter the number of qubits "))
    if qubit > 5 and qubit < 0:
      raise TypeError("Please choose qubit number between 0 and 5")
    if 2**qubit != len(state):
        raise ValueError('You cannot have', len(state) ,'with', qubit, 'qubit(s)')
    qubit_number = qubit

def get_matrix():

    '''
    -> get_matrix()
    -> functionailty: Takes matrix (string in [[],[],...] format) and error (float) as inputs from the user
    -> Return: The Matrix and error
    -> Description: It takes a string written in the mentioned format and error from user, converts the string into an array
    and returns the inputs. We did this so that if a user has a matrix as an output in python, he/she can just can copy and
    paste it in the prompt message or write it as string instead of putting each element one by one> The function also stores
    the matrix in global variable
    '''
    global unitary_matrix
    global error_global
    M = np.array(eval(input("Enter the block encoded matrix in [[],[],..] format: ")))
    if M.shape[0] != 4*qubit_number:
        raise ValueError("The dimension of the block encoded matrix cannot be performed with this numner of qubits")
    err = float(input("Input the tolerance "))
    error_global = err
    unitary_matrix = M

    return M, err

#matrix, err = get_matrix()  To store the input matrix and use it further in the code

def is_square(M):
    '''
    -> is_square()
    -> functionailty: Takes matrix (array) and checks if the number of rows is equal to the number of columns
    -> Return: True/False
    -> Description: It takes the 2D array M and checks if the number of rows ('M.shape[0]') is equal to the number of columns
    (M.shape[1]), if they are equal, it returns 'True', else it returns 'False'
    '''
    return M.shape[0] == M.shape[1] #Check if it's square

def is_unitary(M = None,error = None):
    '''
    -> is_unitary()
    -> functionailty: Takes matrix (array) and checks if it's unitary given the allowed error
    -> Return: True/False
    -> Description: It takes the 2D array M and calculates the transpose conjugate of it (MT) and performs the dot
       product between M and MT. If the dot product is close enough to the identity matrix provided the error
       tolerance then it returns 'True' indicating M is unitary, else it returns 'False'
    '''
    global unitary_matrix
    global product
    if M is None:
        M = unitary_matrix
    if error is None:
        error = error_global
    MT = M.conjugate().T
    prod = np.dot(M, MT)
    #prod_old = prod           #Define a global variable that shows the older product before the adjustment
    for i in range(prod.shape[0]):
        for j in range(prod.shape[1]):
            if i == j:  # Diagonal element
                if abs(prod[i, j] - 1) < error:
                    prod[i, j] = 1
            else:  # Off-diagonal element
                if abs(prod[i, j]) < error:
                    prod[i, j] = 0
    if np.array_equal(prod, np.identity(prod.shape[0])):
        print("The block_encoded matrix is unitary")
    else:
        raise ValueError("The block encoded matrix is not unitary under the threshold, check again!")
    product = prod
    unitary_matrix = M

    return M, prod

def is_square_unitary(M = None, error = None):
    '''
    -> is_square_unitary()
    -> functionailty: Takes matrix (array) and error, checks if it's square and unitary given the allowed error
    -> Return: True/False
    -> Description: This is created specially for testing purpose.

    1. this function takes the arguements matrix M (array) and error (float) if that's provided otherwise it just calls
       get_matrix() i.e. it takes input from the user.
    2. Then it checks if the matrix fulfills the square restriction by calling is_square(), and also checks is the matrix is
       unitary provided the allowed error.

       If both constrains are fulfilled by the matrix, it returns 'True', else 'False'
    '''

    if M is None and error is None:
        M, error = get_matrix()
    if not is_square(M):
        raise AssertionError("Test Failed: Not square")

    if not is_unitary(M,error):
        raise AssertionError("Test Failed: Not unitary")
    return True

def get_hamiltonian_matrix():

    '''
    -> get_hamiltonian_matrix()
    -> functionailty: Takes matrix (string in [[],[],...] format) as input from the user
    -> Return: The Matrix and error
    -> Description: It takes a string written in the mentioned format and error from user, converts the string into an array
    and returns the inputs. We did this so that if a user has a matrix as an output in python, he/she can just can copy and
    paste it in the prompt message or write it as string instead of putting each element one by one
    '''
    global hamiltonian
    H = np.array(eval(input("Enter the Hamiltonian matrix in [[],[],..] format: "))) #converts the string in np.array format and returns as array
    if H.shape[0] != 2*qubit_number:
        raise ValueError("The dimension of the Hamiltonian matrix cannot be performed with this numner of qubits")
    hamiltonian = H

    return H


def is_hermitian(H):
    '''
    -> is_hermitian()
    -> functionailty: Takes matrix (array) and checks if it's hermitian
    -> Return: True/False
    -> Description: It takes the 2D array H and calculates the transpose of it (HT) and checkes if H and it's transpose
       conjugate are same. If they are same then it returns 'True' indicating H is Hermitian, else it returns 'False'
    '''
    return np.array_equal(H, np.conj(H).T)

def normalized_Hamiltonian(H1):
    '''
    -> normalized_Hamiltonian()
    -> functionailty: Takes hamiltonian matrix (array) and returns the normalized matrix
    -> Return: Normalized matrix
    -> Description: It takes the 2D array H and calculates the Frobenius norm of it. It then returns the normalized Hamiltonian
       matrix
    '''
    global normalized_hamiltonian
    H = H1/np.linalg.norm(H1, ord=2)
    normalized_hamiltonian = H
    return H

def is_square_hermitian(H = None):
    '''
    -> is_square_hermitian()
    -> functionailty: Takes matrix (array) and error, checks if it's square, hermitian and the norm is close to 1
    -> Return: True/False
    -> Description: This is created specially for testing purpose.

    1. this function takes the arguements matrix M (array) if that's provided otherwise it just calls get_matrix() i.e. it
       takes input from the user.
    2. Then it checks if the matrix fulfills the square restriction by calling is_square(), and also checks is the matrix is
       unitary provided the allowed error.
    3. It checks if the Frobenius norm is close to 1.

       If all constrains are fulfilled by the matrix, it returns 'True', else 'False'
    '''

    if H is None:
        H = get_hamiltonian_matrix()
    if not is_square(H):
        raise AssertionError("Test Failed: Not square")
    if not is_hermitian(H):
        raise AssertionError("Test Failed: Not hermitian")
    H2 = normalized_Hamiltonian(hamiltonian)
    return True

def is_upper_left(H = None, U = None):

    if H is None:
        H = normalized_hamiltonian
    if U is None:
        U = unitary_matrix

    if H.shape[0] > U.shape[0] or H.shape[1] > U.shape[1]:
        raise AssertionError("The hamiltonian matrix should be smaller than block encoded matrix")
    upper_left_U = U[:H.shape[0],:H.shape[1]]
    if not np.allclose(H, upper_left_U):
        raise AssertionError("The Hamiltonian isn't block-encoded correctly in the upper left of U")

    return True, "The Hamiltonian matrix is in the upper left of U"

def get_hashmap():
      # Prompting the user for the Hamiltonian input
    hashmap_string = input("Enter the Hamiltonian in the format {'gate': weight, ...}: ")
    global hash_map
    # Evaluating the string to convert it to a Python dictionary
    hashmap = eval(hashmap_string)
    hashmap = {key.upper(): value for key, value in hashmap.items()}
    hash_map = hashmap

    return hashmap

def valid_hashmap(hashmap = None):
    valid_gates = {"I", "X", "Y", "Z"}

# convert each input string with .upper()
# then search thru the characters

    if hashmap is None:
        map = get_hashmap()
        hashmap = hash_map
    for term in hashmap.keys():
        for gate in term:
            if gate not in valid_gates:
                raise TypeError("The provided Gate term doesn't belong to Pauli Gates")
    return True

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

paulis = {"I": I, "X": X, "Y": Y, "Z": Z}
map_matrix = None
def hashmap_to_matrix(hashmap=None):
    global map_matrix
    if hashmap is None:
        hashmap = hash_map
    matrix = np.zeros((2**len(list(hashmap.keys())[0]), 2**len(list(hashmap.keys())[0])), dtype=complex)
    for terms, weights in hashmap.items():
        term_matrix = np.array([[1.0]])
        for gate in terms:
            term_matrix = np.kron(term_matrix, paulis[gate])
        matrix += weights * term_matrix
    map_matrix = matrix
    return matrix

def hash_and_hamiltonian(hamiltonian_map=None, hamiltonian_matrix=None):
    if hamiltonian_map is None:
        hamiltonian_map = hashmap_to_matrix()
    if hamiltonian_matrix is None:
        hamiltonian_matrix = hamiltonian
    if not np.allclose(hamiltonian_map, hamiltonian_matrix):
        raise ValueError("Hamiltonian matrix and hash_map don't match, please check!!")
    return True  # Return matrix1 as required

def block_encoded_U(Ham = None):
    global U_block
    if Ham is None:
        Ham = normalized_hamiltonian
    H2=np.matmul(Ham,Ham)
    n = H2.shape[0]

    U = np.zeros([2*n,2*n],dtype=complex)

    I=np.identity(n)
    P = sp.linalg.sqrtm(I-H2)

    U[:n,:n]=Ham
    U[:n,n:]=P
    U[n:,n:]=-Ham
    U[n:,:n]=P

    U_block = U
    return U


def user():
    other_inputs()
    print("If you have the block-encoded matrix, be sure that the norm is an Operator norm")
    Input = input("Do you have the block-encoded matrix? (y/n) ")
    if Input == 'y':
        if not is_square_unitary():
            raise ValueError("Given block encoded matrix is not unitary")
        if not is_square_hermitian():
            raise ValueError("Given Hamiltonian matrix is not hermitian")
        if not is_upper_left():
            raise ValueError("The Hamiltonian is not correctly block encoded")
        if not valid_hashmap():
            raise ValueError("Given hash map is not valid")
        if not hash_and_hamiltonian():
            raise ValueError("Given hash map and the Hamiltonian matrix don't match")
    elif Input == 'n':
        Input_2 = input("Do you have the hamiltonian matrix? (y/n) ")
        if Input_2 == 'y':
            if not is_square_hermitian():
                raise ValueError("Given Hamiltonian matrix is not hermitian")
            if not valid_hashmap():
                raise ValueError("Given hash map is not valid")
            if not hash_and_hamiltonian():
                raise ValueError("Given hash map and the Hamiltonian matrix don't match")
            normalized_Hamiltonian(map_matrix)
            block_encoded_U()
        elif Input_2 == 'n':
            if not valid_hashmap():
                raise ValueError("Given hashmap is not valid")
            hashmap_to_matrix()
            normalized_Hamiltonian(map_matrix)
            block_encoded_U()
        else:
            raise TypeError("Invalid input")
    else:
        raise TypeError("Invalid input")
    return True

if __name__ == "__main__":
    user()
    output_info_list = list(eval(input('Provide a list of keys to perform desired tasks based on the following legend:\n0: statevector at time t\n1: energy at time t\n2: Energy evolution data up to max time\n3: fidelity evolution data up to max time\n\t\t: ')))

    for item in output_info_list:
        if item == 0:
            sv_file_name = input('Provide a file name for your state vector data: ')
            sv_data = ['Numerical statevector at time {}: \n{}'.format(execution_time, hs.NumericalHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).getStatevector(execution_time=execution_time, initial_state=initial_state)),
                       'Trotter statevector at time {}: \n{}'.format(execution_time, hs.TrotterHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).getStatevector(execution_time=execution_time, initial_state=initial_state)),
                       "QSP statevector at time {}: \n{}".format(execution_time, run_HamSim_H(num_qubits=qubit_number, H = normalized_hamiltonian, evolution_time=execution_time)[1].data),
                       'Simulation Type: {}'.format('statevector'),
                       'Hamiltonian Map: {}'.format(hash_map),
                       'Qubit Count: {}'.format(qubit_number),
                       'Execution Time: 0 to {}'.format(execution_time),
                       'Initial State: {}'.format(initial_state)]
            file_sv_data = "statevector_{}_metadata.txt".format(sv_file_name)
            with open(file_sv_data, mode = "w") as f:
                for item in sv_data:
                    f.write(item + '\n')
        if item == 1:
            energy_file_name = input('Provide a file name for your energy data: ')
            energy_data = ['Numerical energy at time {}: \n{}'.format(execution_time, hs.NumericalHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).getEnergy(execution_time=execution_time, initial_state=initial_state)),
                           'Trotter energy at time {}: \n{}'.format(execution_time, hs.TrotterHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).getEnergy(execution_time=execution_time, initial_state=initial_state)),
                           'Simulation Type: {}'.format('energy'),
                           'Hamiltonian Map: {}'.format(hash_map),
                           'Qubit Count: {}'.format(qubit_number),
                           'Execution Time: 0 to {}'.format(execution_time),
                           'Initial State: {}'.format(initial_state)]
            file_energy_data = "energy_{}_metadata.txt".format(energy_file_name)
            with open(file_energy_data, mode = "w") as f:
                for item in energy_data:
                    f.write(item + '\n')
        if item == 2:
            input_name_sv = input('Provide a file name for your energy evolution data file: ')
            hs.NumericalHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).energyEvolutionData(file_name = input_name_sv, initial_state=initial_state, max_time=execution_time)
            hs.TrotterHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).energyEvolutionData(file_name = input_name_sv, initial_state=initial_state, max_time=execution_time)
        if item == 3:
            input_name_fid = input('Provide a file name for your fidelity evolution data file: ')
            hs.TrotterHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).NTFidelityTest(file_name = input_name_fid, initial_state=initial_state, max_time=execution_time)

    qc_file_name = input('Provide a file name for your quantum circuit data: ')
    hs.TrotterHamiltonianSimulation(qubit_count=qubit_number, pauli_hash_map=hash_map).getOpenQASM(file_name=qc_file_name)