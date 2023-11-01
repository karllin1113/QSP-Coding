# Hamiltonian Simulation Coding Project
This coding project is aimed at conducting Hamiltonian simulaiton via three mehtods:
1. Numerical (classical) calculation
2. Trottersiation Method
3. Quantum Signal Processing Algorithm

## Installation
First, create and active a Python virtual environment:
```
conda create -n <name> python=3.9
conda active <name>
```
Then, pip install the required dependencies:
```
pip install -r requirement.txt
```
Next, clone the repository:
```
git clone https://...
```
## Running the file
To run the file, move into the correct directory then call the file ```parent.py```:
```
cd <folder name>
python parent.py
```

## Taking User Inputs
If you have read and understood the documentation for the modules, the user inputs should be fairly trivial. The example we will work with is a trivial one.
Our input Hamiltonian is given by the hash_map = {'xx':1}. (Assume we do not know the Hamiltonian matrix explicitly)
Our execution time is 3.14
We do not have a block encoded matrix. We do not have the Hamiltonian matrix.
Our desired outputs are option '0' - statevector at time t

The first of a series of questions prompts:
1. *Enter the initial state in [...] format:*
```
Enter the initial state in [...] format: [1,1,1,1]
```

2. *Enter the time between 0 and 5 to execute*
```
Enter the time between 0 and 5 to execute : 3.14
```

3. *Enter the number of qubits*
```
Enter the number of qubits 2
```

4. *Do you have the block-encoded matrix? (y/n)*
```
Do you have the block-encoded matrix? (y/n) n
```

5. *Do you have the hamiltonian matrix? (y/n)*
```
Do you have the hamiltonian matrix? (y/n) n
```

6. *Provide a list of keys to perform desired tasks based on the following legend:*
    *0: statevector at time t*
    *1: energy at time t*
    *2: Energy evolution data up to max time*
    *3: fidelity evolution data up to max time*
                    *:*
```
Provide a list of keys to perform desired tasks based on the following legend:
0: statevector at time t
1: energy at time t
2: Energy evolution data up to max time
3: fidelity evolution data up to max time
                :[0]
```

7. *Questions*

8. *Provide a file name for your quantum circuit data:*
```
Provide a file name for your quantum circuit data: demo_qc
```
### Output
By default, the Trotter and QSP ```QISKIT``` OpenQASM files will be saved.

Based on tasks requested at question 6, a number of files will be saved also.

- 0: metadata.txt file with statevector information
- 1: metadata.txt file with energy informaiton
- 2: metadata.txt file for the problem setting and a data.csv file with the energy evolution data for the spread of times
- 3: metadata.txt file for the problem setting and a data.csv file with the fidelity evolution data for the spread of times

If '[]' is passed at question 6 then only the quantum circuit for the problem setting is saved.
## Testing
To test the numerical and Trotter Hamiltonian simulation module, run:
```
python tests/numerical_trotter_testing.py
```
## Contributions
This code has three main contributors. The main contributions of each author is as follows (not exhaustive contribution list):
- Gabriel Waite: numerical and Trotter Hamiltonian simulation code
- ...
