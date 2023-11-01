import numpy as np
from numpy.linalg import norm, eigvals, eig

from qiskit import QuantumCircuit, qasm
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity

import time
import json
from sympy import Matrix
import matplotlib.pyplot as plt

from .helper_functions import construct_BE_rand, construct_BE, convert_binary_to_int
from .HamiltonianSimulation_by_QET import HamSim_byQET


I = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])


def run_HamSim_H(num_qubits: int,
                H: np.ndarray,
                evolution_time: float = 5.0,
                truncation_order: int = 8,
                starting_state: np.ndarray = None,
                ) -> (Statevector, Statevector):

    
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)

    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                              BE_H = BE_H.data,
                              evolution_time = evolution_time,
                              truncation_order = truncation_order,
                              error_tolerance=1e-6,
                              starting_state = None)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    # QSP_circ = obj_HS_QET.getCirc()

    output_state, output_state_reversed = obj_HS_QET.runHamiltonianSimulator()
    # QSP_circ.qasm(filename="QSP_circuit")


    return output_state, output_state_reversed

"""
if __name__ == "__main__":

    ## Input Parameters
    num_qubits = 2
    evolution_time = 3.0
    truncation_order = 8
    starting_state = None
    
    # H = np.kron(np.kron(Y, X), Z)
    # H = np.kron(np.kron(np.kron(np.kron(Y, X), Z), Z), Y) ## takes 2 mins
    H = np.kron(Y, X) + np.kron(Z, Z) # + np.kron(X, Y)
    ## H = X + Y ## Good
    ## H = Z + X or Z + Y won't even compile
    ## H = X + Z
    ## H = np.kron(Z, Z) +  np.kron(X, I) + np.kron(I, X)
    # H = Z


    fidelity, bench_state, output_state = run_HamSim_H(num_qubits = num_qubits,
                                                       H = H,
                                                       evolution_time = evolution_time,
                                                       truncation_order = truncation_order,
                                                       starting_state = starting_state)

    print("Fidelity", fidelity)
    output_state.draw("latex")
    Matrix(output_state.data)

"""


    # initial_state = Statevector.from_int(0, 2 ** num_qubits)

    # first_dict = {
    #     "time": 0.0,
    #     "fidelity": 1.0,
    #     "real_QSP": initial_state.data.real.tolist(),
    #     "imag_QSP": initial_state.data.imag.tolist(),
    #     "real_Truth": initial_state.data.real.tolist(),
    #     "imag_Truth": initial_state.data.imag.tolist()
    # }


    # with open(json_filename, "w") as openfile:
    #     json.dump([first_dict], openfile)


    # json_list = []
    # count = 1

    # for _t in np.linspace(0, max_evolution_time, 50)[1:]:
        
    #     count += 1
    #     time_start = time.perf_counter()
        
    #     fidelity, bench_state, output_state = run_HamSim_H(num_qubits, H = H,
    #                                                         evolution_time=_t, truncation_order = truncation_order)
    #     time_end = time.perf_counter()
        
    #     print("Count {}: at t = {:.4f}, Fidelity between bench_state and qet_state = {}".format(count, _t, fidelity))
    #     print("Time took:", time_end - time_start)
        
        
    #     real_QSP = output_state.data.real.tolist()
    #     imag_QSP = output_state.data.imag.tolist()

    #     real_Truth = bench_state.data.real.tolist()
    #     imag_Truth = bench_state.data.imag.tolist()
        
    #     curr_dict = {
    #         "time": _t,
    #         "fidelity": fidelity,
    #         "real_QSP": real_QSP,
    #         "imag_QSP": imag_QSP,
    #         "real_Truth": real_Truth,
    #         "imag_Truth": imag_Truth
    #     }

    #     json_list.append(curr_dict)

    #     if count % 5 == 0:
    #         with open(json_filename, "r") as openfile:
    #             json_object = json.load(openfile)

    #         json_object.extend(json_list)

    #         with open(json_filename, "w") as openfile:
    #             json.dump(json_object, openfile)

    #         json_list = []


    # if len(json_list) != 0:
    #     print("Last dicts to be stored!")
        
    #     with open(json_filename, "r") as openfile:
    #         json_object = json.load(openfile)

    #     json_object.extend(json_list)

    #     with open(json_filename, "w") as openfile:
    #         json.dump(json_object, openfile)