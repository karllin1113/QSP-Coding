import numpy as np
from math import factorial
import json

from numpy.linalg import norm, eigvals, eig
from numpy.polynomial.chebyshev import Chebyshev, cheb2poly

from scipy.linalg import sqrtm
from scipy.special import jv

from qiskit.quantum_info import Operator, random_hermitian



class PostMsmtError(Exception):
    """ The error raised when 1000 rounds of post selection cannot produce the desired msmt outcome (all 0's state)"""
    pass

class QSPGenerationError(Exception):
    """ The error raised when 5000 calls to I. Chuang et al.'s PyQSP package cannot generate the correct QSP phase angles of the given parameters."""
    pass

class BlockEncodingError(Exception):
    """ The error raised when the block encoding of the input Hermitian matrix cannot be constructed. """
    pass



def construct_BE(num_qubits: int,
                 H: np.ndarray,
                 num_BE_qubits: int = 1
                ) -> (Operator, Operator):

    if (num_qubits != int(np.log2(H.shape[0]))) or (num_qubits != int(np.log2(H.shape[1]))):
        raise BlockEncodingError("The dimension of H and num_qubits do not match.")
    
    ## Normalize the Hermitian matrix
    ##  For whatever reason, H needs to be normalised twice before it can be block encoded.
    H = H / norm(H, 2)
    H = H / norm(H, 2)
    
    ## H = H / (norm(H, 2)*1.05)
    
    for _ in range(100): 

        # print("Round: {}. Norm = {}".format(_, norm(H, 2)))
        
        off_diag = sqrtm(np.identity(2 ** num_qubits) - H @ H)

        BE_H = np.kron(np.array([[1., 0.], [0., 0.]]), H) + np.kron(np.array([[0., 1.], [0., 0.]]), off_diag) + np.kron(np.array([[0., 0.], [1., 0.]]), off_diag) - np.kron(np.array([[0., 0.], [0., 1.]]), H)
        
        if Operator(BE_H).is_unitary():
            ## return Operator(BE_H)
            return Operator(H), Operator(BE_H)

    raise BlockEncodingError("Block encoding of the input Hermitian matrix cannot be constructed.")


def construct_BE_rand(num_qubits: int,
                      num_BE_qubits: int = 1
                      ) -> (Operator, Operator):

    ## It takes some time to run for larger num_qubits
    
    for _ in range(1000):
        H = random_hermitian(2 ** num_qubits).data
        H = H / norm(H, 2)
        ## H = H / (norm(H, 2)*1.05)
    
        off_diag = sqrtm(np.identity(2 ** num_qubits) - H @ H)

        BE_H = np.kron(np.array([[1., 0.], [0., 0.]]), H) + np.kron(np.array([[0., 1.], [0., 0.]]), off_diag) + np.kron(np.array([[0., 0.], [1., 0.]]), off_diag) - np.kron(np.array([[0., 0.], [0., 1.]]), H)
        
        if Operator(BE_H).is_unitary():
            return Operator(H), Operator(BE_H)

    raise BlockEncodingError("Block encoding of the input Hermitian matrix cannot be constructed.")



# def check_if_properBE(num_qubits: int,
#                       BE_H: np.ndarray,
#                       num_BE_qubits: int = 1):

#         # Check if BE_H is unitary
#         if not Operator(BE_H).is_unitary():
#             print("ERROR: Input np.ndarray BE_H is not unitary")
#             return False
    
#         # Check if n+m = BE_H.dimen
#         BE_num_qubits = int(np.log2(BE_H.shape[0]))
#         if not (num_qubits + num_BE_qubits) == BE_num_qubits:
#             print("ERROR: Num of qubits of the Hamiltonian + num of Block-encoding qubits does not match with the number of qubits of BE_H")
#             return False
    
#         return True


def cos_xt_taylor(t: float = 1.0,
                  d: int = 20,
                 ) -> (np.ndarray, str):
    
    coeffs = [1]
    string_exp = "1"
    
    for i in range(1, d+1):
        _coeff = (-1)**i / factorial(2*i) * t**(2*i)
        coeffs.extend([0, _coeff])
        string_exp += " + {}*x**{}".format(_coeff, 2*i)
        
    # return 2*d+1 coeffs, d+1 terms
    return np.array(coeffs), string_exp


def sin_xt_taylor(t: float = 1.0,
                  d: int = 20,
                 ) -> (np.ndarray, str):
    
    coeffs=[]
    string_exp = ""
    
    for i in range(d+1):
        _coeff = (-1)**i / factorial(2*i + 1) * t**(2*i + 1)
        coeffs.extend([0, _coeff])
        string_exp += " + {}*x**{}".format(_coeff, 2*i + 1)

    # return 2*d+2 coeffs, d+1 terms
    return np.array(coeffs), string_exp


def cos_xt_JA(t: float = 1.0,
             d: int = 20
             ) -> (np.ndarray, str):

    cos_bessel_coeffs = [jv(0, t)]
    for i in range(1, d+1):
        cos_bessel_coeffs += [0, 2 * (-1)**i * jv(2*i, t)]

    cos_power_series = cheb2poly(cos_bessel_coeffs)

    cos_string_poly = ""
    for i in range(len(cos_power_series)):
        cos_string_poly += "+ {} * x**{}".format(cos_power_series[i], i)
    
    return cos_power_series, cos_string_poly


def sin_xt_JA(t: float = 1.0,
             d: int = 20
             ) -> (np.ndarray, str):

    sin_bessel_coeffs = []
    for i in range(d+1):
        sin_bessel_coeffs += [0, 2 * (-1)**i * jv(2*i+1, t)]

    sin_power_series = cheb2poly(sin_bessel_coeffs)

    sin_string_poly = ""
    for i in range(len(sin_power_series)):
        sin_string_poly += "+ {} * x**{}".format(sin_power_series[i], i)
    
    return sin_power_series, sin_string_poly


def convert_binary_to_int(array):
    value = 0
    for i in range(len(array)):
        value += 2 ** i * array[len(array)-1 -i]

    return value