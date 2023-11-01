import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import BasicAer
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, partial_trace, state_fidelity, random_hermitian, random_unitary
from qiskit.extensions import UnitaryGate
from qiskit.visualization import plot_state_city

from .pyqsp_master.pyqsp.angle_sequence import QuantumSignalProcessingPhases
from .pyqsp_master.pyqsp.response import PlotQSPResponse, PlotQSPPhases

from .helper_functions import cos_xt_taylor, sin_xt_taylor, cos_xt_JA, sin_xt_JA
from .helper_functions import PostMsmtError, QSPGenerationError, BlockEncodingError



class SignalProcessingOperators_OneBEQubit():

    def __init__(self,
                ang_list: list):

        if not isinstance(ang_list, list):
            raise ValueError("Error Msg")

        self.phase_ops = []
        self.phase_ops_op = []
        self.ang_list = ang_list

    def buildCircs(self):

        if not len(self.phase_ops) == 0:
            print("WARNING MSG: self.phase_ops is not empty - overwriting the old circuits now")
            
        master_list_ops = []
        master_list_ops_op = []
        
        for phi in self.ang_list:        
            circ = QuantumCircuit(2)
            circ.cx(1, 0, ctrl_state=0)
            ### =============================================================================================
            ### Should be 2*phi according to ComputeQSPResponse().qsp_op(phi) (Ln: 42). What about +/-??
            ### https://qiskit.org/documentation/stubs/qiskit.circuit.library.RZGate.html#qiskit.circuit.library.RZGate
            # circ.rz(phi, 0)
            ### For QSP (according to PyQSP)
            # circ.rz(-2 * phi, 0)
            circ.rz(2*phi, 0)
            # circ.append(Operator(np.array([[np.exp(- 1j * phi), 0.], [0., np.exp(1j * phi)]])), [0])
            ### =============================================================================================
            circ.cx(1, 0, ctrl_state=0)
            
            master_list_ops.append(circ)
            master_list_ops_op.append(Operator(circ))

        self.phase_ops = master_list_ops
        self.phase_ops_op = master_list_ops_op

    def getCircs(self) -> list:
        return self.phase_ops
    
    def getOps(self) -> list:
        return self.phase_ops_op
    



class QET_LinLin():

    # ===========================================================================
    def _check_if_properBE(self,
                          num_qubits: int,
                          num_BE_qubits: int,
                          BE_H: np.ndarray):

        # Check if BE_H is unitary
        if not Operator(BE_H).is_unitary():
            print("ERROR: Input np.ndarray BE_H is not unitary")
            return False
    
        # Check if n+m = BE_H.dimen
        BE_num_qubits = int(np.log2(BE_H.shape[0]))
        if not (num_qubits + num_BE_qubits) == BE_num_qubits:
            print("ERROR: Num of qubits of the Hamiltonian + num of Block-encoding qubits does not match with the number of qubits of BE_H")
            return False
    
        return True
    # ===========================================================================

    
    def __init__(self,
                 num_qubits: int,
                 # TODO: Should BE_H be Gate, QuantumCircuit, np.ndarray, or others?
                 BE_H: np.ndarray,
                 # TODO:
                 phase_angles: list,
                 num_BE_qubits: int = 1,
                 phase_angles_order: str = "desc",   # {"desc", "asce"}
                 real_poly: bool = True,
                 special_case: bool = False,
                 # style: str = "linlin",              # {"Linlin", "grand_uni"}
                ):
        # TODO: reorder parameters and clarify their descrips. 
        """
        Input:
        1. phase_angles
        type: python list
        Ordered in Lin Lin's fashion

        3. num_qubits
        type: int
        Number of qubits of the Hamiltonian

        # TODO Should BE_H be Gate, QuantumCircuit, np.ndarray, or others?
        4. BE_H
        type: np.ndarray
        Block encoding unitary of the Hamiltonian H

        4. num_BE_qubits
        type: int, default value = 1
        The number of qubits that block encode the Hamiltonian H

        2. phase_angles_order
        type: str, default value = "desc"
        Order of the phase angles, either "desc" or "asce"
        
        5. style
        type: str, default value = "linlin"
        Implement the QET circuit in either (1) Lin Lin or (2) Grand Uni's fashion
        """
        # ======================================================================================
        # TODO: chech parameters that are later added, i.e. num_BE_qubits
        # TODO: check if n + m = qubits of BE_H
        if not isinstance(phase_angles, list):
            raise ValueError("Error message #1")
        if not isinstance(num_qubits, int):
            raise ValueError("Error message #2")
        # TODO Should BE_H be Gate, QuantumCircuit, np.ndarray, or others?
        if not isinstance(BE_H, np.ndarray):
            raise ValueError("Error message #3")
        # TODO
        if not isinstance(phase_angles_order, str):
            pass
        # TODO
        # if not (isinstance(style, str)):
        #    pass
        # ======================================================================================

        
        # ======================================================================================
        # Check if BE is indeed a correct block encoding?
        # maybe write a fun check_BE(): and use e.g. .is_unitary() outside the class and call it here
        # 
        # or place a function at higher level and call it before calling this class()
        
        if not self._check_if_properBE(num_qubits, num_BE_qubits, BE_H):
            raise ValueError("ERROR MSG: BE_H is not a proper block encoding.")
        else:
            # print("Verified: BE_H is a proper block encoding. Program continues.")
            pass            
        
        # ======================================================================================
        
        self.num_qubits = num_qubits
        self.num_BE_qubits = num_BE_qubits
        self.BE_H = BE_H
        
        if phase_angles_order == "asce":
            phase_angles.reverse()
        self.phase_angles = phase_angles
        self.phase_angles_order = phase_angles_order

        self.real_poly = real_poly
        self.special_case = special_case

        self.d = len(phase_angles) - 1
        self.BE_Operator = None
        self.obj_phase = None
        self.main_circ = None
        self.main_circ_op = None

    def buildCirc(self):

        if self.main_circ is not None:
            print("WARNING MSG: self.main_circ is not None - overwriting the old QET circuit now")
            self.main_circ = None
            self.main_circ_op = None
        
        # Quantum registers
        qreg_ctrl = QuantumRegister(1, "signal_ctrl")
        qreg_be = QuantumRegister(self.num_BE_qubits, "BE")
        qreg_input = QuantumRegister(self.num_qubits, "input")
                        
        qet_circ = QuantumCircuit(qreg_ctrl, qreg_be, qreg_input)

        # Build an qiskit.quantum_info.Operator instance from BE_H 
        self.BE_Operator = Operator(self.BE_H)
        
        # Build the signal processing operators
        self.obj_phase = SignalProcessingOperators_OneBEQubit(self.phase_angles)
        self.obj_phase.buildCircs()


        # print("Right before the if statements. d now is of type {}, and of value {}".format(type(self.d), self.d))
        
        # TODO
        # when d is even
        if self.d % 2 == 0 and self.special_case is False:

            # print("d = {} is even. The theorem of QET for even d implemented.".format(self.d))
            
            # Since the order of the phase angles are descending, we need to transverse the list of angles reversely
            # NB: d/2 is float

            ### =====================================================================
            # Lin Lin Figure 7.9
            if self.real_poly:
                qet_circ.h(qreg_ctrl)
            ### =====================================================================
            
            for i in range(self.d//2, 0, -1):
                qet_circ.append(self.obj_phase.getOps()[2*i],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                # For whatever reason, 
                # qet_circ.append(self.BE_Operator, [qreg_be, qreg_input]) does not work
                
                # qet_circ.append(self.BE_Operator,
                #                 [qreg_be[i] for i in range(self.num_BE_qubits)] + [qreg_input[i] for i in range(self.num_qubits)])

                qet_circ.append(self.BE_Operator,
                                [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])
                
                qet_circ.append(self.obj_phase.getOps()[2*i - 1],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                # qet_circ.append(self.BE_Operator.adjoint(),
                #                 [qreg_be[i] for i in range(self.num_BE_qubits)] + [qreg_input[i] for i in range(self.num_qubits)])

                qet_circ.append(self.BE_Operator.adjoint(),
                                [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])
    
            qet_circ.append(self.obj_phase.getOps()[0],
                            [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])

            ### =====================================================================
            # Lin Lin Figure 7.9
            if self.real_poly:
                qet_circ.h(qreg_ctrl)
            ### =====================================================================

            self.main_circ = qet_circ
            self.main_circ_op = Operator(qet_circ)
            
            return
            
        # when d is odd
        elif self.d % 2 == 1 and self.special_case is False:

            # print("d = {} is odd. The theorem of QET for odd d implemented.".format(self.d))

            ### =====================================================================
            # Lin Lin Figure 7.9
            if self.real_poly:
               qet_circ.h(qreg_ctrl)
            ### =====================================================================         
            
            # Since the order of the phase angles are descending, we need to transverse the list of angles reversely
            # NB: d/2 is float    
            for i in range((self.d - 1)//2, 0, -1):
                qet_circ.append(self.obj_phase.getOps()[2*i + 1],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                # For whatever reason, 
                # qet_circ.append(self.BE_Operator, [qreg_be, qreg_input]) does not work
                
                # qet_circ.append(self.BE_Operator, 
                #                 [qreg_be[i] for i in range(self.num_BE_qubits)] + [qreg_input[i] for i in range(self.num_qubits)])

                qet_circ.append(self.BE_Operator, 
                                [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])
                
                qet_circ.append(self.obj_phase.getOps()[2*i],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                
                # qet_circ.append(self.BE_Operator.adjoint(), 
                #                 [qreg_be[i] for i in range(self.num_BE_qubits)] + [qreg_input[i] for i in range(self.num_qubits)])

                qet_circ.append(self.BE_Operator.adjoint(), 
                               [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])

            qet_circ.append(self.obj_phase.getOps()[1],
                            [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
            
            # qet_circ.append(self.BE_Operator, 
            #                 [qreg_be[i] for i in range(self.num_BE_qubits)] + [qreg_input[i] for i in range(self.num_qubits)])

            qet_circ.append(self.BE_Operator, 
                            [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])
    
            qet_circ.append(self.obj_phase.getOps()[0],
                            [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])

            
            ### =====================================================================
            ## Lin Lin's constant p.108
            # qet_circ.append(Operator((0-1j)**self.d * np.identity(2 ** (1 + self.num_BE_qubits + self.num_qubits))),
            #                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)] + [qreg_input[i] for i in range(self.num_qubits)])

            # qet_circ.append(Operator((0-1j)**self.d * np.identity(2 ** (1 + self.num_BE_qubits + self.num_qubits))),
            #               [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)] + [qreg_ctrl])
            ### =====================================================================
            
            ### =====================================================================
            # Lin Lin Figure 7.9
            if self.real_poly:
               qet_circ.h(qreg_ctrl)
            ### =====================================================================
            
            self.main_circ = qet_circ
            self.main_circ_op = Operator(qet_circ)
            return 
        
        ## Gabe's modification
        elif self.special_case is True:
            # So far this only works for 1 BE qubit
            if self.num_BE_qubits > 1:
                raise ValueError('We cant do more than one RN!')
            else:
                # Note: ihave changed the size of the smaller elements from num_qb to BE_num_qb + num_qb.
                # There was an error rasied when i appeneded to 1 + self.num_BE_qubits + self.num_qubits
                big_X_block_size = 2 ** (self.num_qubits)
                big_X_matrix = np.block([
                    [np.zeros((big_X_block_size, big_X_block_size)), np.eye(big_X_block_size)],
                    [np.eye(big_X_block_size), np.zeros((big_X_block_size, big_X_block_size))]])
                
            big_X = UnitaryGate(Operator(big_X_matrix), label = 'Big X')
 
            qet_circ.append(big_X, [i for i in range(1,1+self.num_BE_qubits + self.num_qubits)])
 
            self.main_circ = qet_circ
            # self.main_circ_op = Operator(qet_circ)
            return
            


            
    def getCirc(self):
        if self.main_circ is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
        return self.main_circ

    def getOp(self):
        if self.main_circ_op is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
        return self.main_circ_op
        

    # May delete this
    def _getDecomposedCirc(self):
        if self.main_circ is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
            return None
        return self.main_circ.decompose() 
    
    # TODO: .draw("mpl") does not work here
    def drawCirc(self,
                output: str = "text",
                decomp: bool = False):
        if self.main_circ is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
            return None

        if decomp:
            print(self.main_circ.decompose().draw(output=output))
        else:
            print(self.main_circ.draw(output=output))

    # def runCirc(self):
    #     pass

    # def post_selection(self):
    #     # add msmt and post-select for the all-0s state
    #     pass




class HamSim_byQET():

    # ===========================================================================
    def _check_if_properBE(self,
                          num_qubits: int,
                          num_BE_qubits: int,
                          BE_H: np.ndarray):

        # Check if BE_H is unitary
        if not Operator(BE_H).is_unitary():
            print("ERROR: Input np.ndarray BE_H is not unitary")
            return False
    
        # Check if n+m = BE_H.dimen
        BE_num_qubits = int(np.log2(BE_H.shape[0]))
        if not (num_qubits + num_BE_qubits) == BE_num_qubits:
            print("ERROR: Num of qubits of the Hamiltonian + num of Block-encoding qubits does not match with the number of qubits of BE_H")
            return False
    
        return True
    # ===========================================================================
    
    def __init__(self,
                num_qubits: int,
                BE_H: np.ndarray,
                evolution_time: float,
                num_BE_qubits: int = 1,
                truncation_order: int = 8,              ## truncation order = 10, 11
                approx_method: str = "Jacobi-Anger",
                error_tolerance: float = 1e-6,          ## PyQSP's default: 1e-6
                starting_state: np.ndarray = None,
                simulator_BasicAer: bool = False,
                ):
        """
        class description
        """
        
        # ================================================
        # TODO: 
        # isinstance()
        #
        #
        #
        # ================================================

        # ======================================================================================
        # Check if BE is indeed a correct block encoding?
        # maybe write a fun check_BE(): and use e.g. .is_unitary() outside the class and call it here
        # 
        # or place a function at higher level and call it before calling this class()
        
        if not self._check_if_properBE(num_qubits, num_BE_qubits, BE_H):
            raise ValueError("ERROR MSG: BE_H is not a proper block encoding.")
        else:
            # print("Verified: BE_H is a proper block encoding. Program continues.")
            pass            
        
        # ======================================================================================
        
        self.num_qubits = num_qubits
        self.num_BE_qubits = num_BE_qubits
        self.BE_H = BE_H

        self.evolution_time = evolution_time
        self.truncation_order = truncation_order
        self.approx_method = approx_method
        
        self.error_tolerance = error_tolerance
        
        ### checking if starting state has the correct number of qubits
        if starting_state is not None:
            if len(starting_state) != 2 ** self.num_qubits:
                raise ValueError("Initial state has to be of size of {} qubits".format(2 ** self.num_qubits))

        # if starting_state is not None:
        #     if len(starting_state) != self.num_qubits:
        #         raise ValueError("Initial state has to be of size of {} qubits".format(self.num_qubits))
            
        self.starting_state = starting_state
        self.simulator_BasicAer = simulator_BasicAer

        # ===========================================
        self.cos_coeffs = None
        self.cos_ang_seq = None
        
        self.sin_coeffs = None
        self.sin_ang_seq = None

        self.obj_QET_cos = None
        self.obj_QET_sin = None
        
        self.HamSim_circ = None

    
    def computeQSPPhaseAngles(self):

        """
        if (self.cos_ang_seq is not None) or (self.sin_ang_seq is not None):
            print("WARNING MSG: self.cos_ang_seq or self.sin_ang_seq is not None - overwriting the old QSP phase angles now.")
            self.cos_ang_seq = None
            self.sin_ang_seq = None
        """

        if (self.cos_ang_seq is not None) and (self.sin_ang_seq is not None):
            print("QSP phase angles have been generated. Returning None now.")
            return

        if self.approx_method == "Taylor":
            self.cos_coeffs, _ = cos_xt_taylor(t = self.evolution_time,
                                               d = self.truncation_order)
        
            self.sin_coeffs, _ = sin_xt_taylor(t = self.evolution_time,
                                               d = self.truncation_order)
            # print("Taylor expansions for cos(xt) and sin(xt) truncated at order {} and {}, respectively, for t = {}."\
            #       .format(2*self.truncation_order, 2*self.truncation_order+1, self.evolution_time))

        
        elif self.approx_method == "Jacobi-Anger":
            self.cos_coeffs, _ = cos_xt_JA(t = self.evolution_time,
                                           d = self.truncation_order)
        
            self.sin_coeffs, _ = sin_xt_JA(t = self.evolution_time,
                                           d = self.truncation_order)
            # print("Jacobi-Anger expansions for cos(xt) and sin(xt) truncated at order {} and {}, respectively, for t = {}."\
            #       .format(2*self.truncation_order, 2*self.truncation_order+1, self.evolution_time))

#         print("""Parameters for computing QSP angles:
# 1. evolution time = {}
# 2. approximation method = {}
# 3. truncation order = {} (for cos(xt)), {} (for sin(xt))
# 4. QSP angles error tolerance = {}\n\nComputing...\n""".format(self.evolution_time, self.approx_method, 2*self.truncation_order, 2*self.truncation_order+1, self.error_tolerance))
        
        
        for _ in range(5000):
            try:
                if self.evolution_time > 0:
                    self.cos_ang_seq = QuantumSignalProcessingPhases(self.cos_coeffs, signal_operator="Wx", tolerance=self.error_tolerance)
                    self.sin_ang_seq = QuantumSignalProcessingPhases(self.sin_coeffs, signal_operator="Wx", tolerance=self.error_tolerance)
                
                ## Gabe's modification
                else:
                    self.cos_ang_seq, self.sin_ang_seq = [0], [None]    ## Should be None
            
            except:
                self.cos_ang_seq = None
                self.sin_ang_seq = None
                continue
            else:
                # print("QSP phase angles for cos(xt) and sin(xt) successfully generated after {} rounds!".format(_+1))
                return

        raise QSPGenerationError("""FAILURE: QSP phase angles failed to be generated after 5000 tries.
Methods for finding QSP angles become numerically unstable when the evolution time or truncation order is too high. 
We recommand reducing the evolution time to be less than 5.0 and the truncation order to less than or equal to 5.""")



    
    def plotQSPPhaseAngles(self):
        if (self.cos_ang_seq is None) or (self.sin_ang_seq is None):
            print("QSP phase angles haven't been computed yet. To compute the angles, please call self.computeQSPPhaseAngles(). Returing None now.")
            return None

        print("QSP phase angles for cos(xt) when t = {} are\n{}\n".format(self.evolution_time, self.cos_ang_seq))
        print("The plot for cos(xt) phase angles:")
        PlotQSPPhases(self.cos_ang_seq)
        print("The plot for QSP response function for cos(xt):")
        PlotQSPResponse(self.cos_ang_seq, target= lambda x: np.cos(x * self.evolution_time), signal_operator="Wx")

        print("QSP phase angles for sin(xt) when t = {} are\n{}\n".format(self.evolution_time, self.sin_ang_seq))
        print("The plot for sin(xt) phase angles:")
        PlotQSPPhases(self.sin_ang_seq)
        print("The plot for QSP response function for sin(xt):")
        PlotQSPResponse(self.sin_ang_seq, target= lambda x: np.sin(x * self.evolution_time), signal_operator="Wx")

    
    def buildCirc(self):
        if (self.cos_ang_seq is None) or (self.sin_ang_seq is None):
            print("QSP phase angles haven't been computed yet. To compute the angles, please call self.computeQSPPhaseAngles(). Returing None now.")
            return None
        if self.HamSim_circ is not None:
            print("WARNING MSG: self.main_circ is not None - overwriting the old QET circuit now")
            self.HamSim_circ = None
        
        # Quantum registers
        qreg_HS_ctrl = QuantumRegister(1, "HamSim_ctrl")
        qreg_QET_input = QuantumRegister(1 + self.num_BE_qubits + self.num_qubits)

        main_circ = QuantumCircuit(qreg_HS_ctrl, qreg_QET_input)

        # calling the QET_LinLin() class twice
        obj_QET_cos = QET_LinLin(num_qubits = self.num_qubits,
                                 BE_H = self.BE_H,
                                 phase_angles = self.cos_ang_seq,
                                 num_BE_qubits = self.num_BE_qubits)
        
        if self.sin_ang_seq == [None]:
            obj_QET_sin = QET_LinLin(num_qubits = self.num_qubits,
                                    BE_H = self.BE_H,
                                    phase_angles = self.sin_ang_seq,
                                    num_BE_qubits = self.num_BE_qubits,
                                    special_case = True)
        else:
            obj_QET_sin = QET_LinLin(num_qubits = self.num_qubits,
                                    BE_H = self.BE_H,
                                    phase_angles = self.sin_ang_seq,
                                    num_BE_qubits = self.num_BE_qubits)

        
        # Saving the two objects                        
        self.obj_QET_cos = obj_QET_cos
        self.obj_QET_sin = obj_QET_sin

        obj_QET_cos.buildCirc()
        obj_QET_sin.buildCirc()
        
        # obj_QET_cos.drawCirc()
        # obj_QET_sin.drawCirc()
        
        # ControlledGate
        # https://qiskit.org/documentation/stubs/qiskit.circuit.ControlledGate.html
        ctrl_op_cos = obj_QET_cos.getCirc().to_gate().control(num_ctrl_qubits=1, ctrl_state="0")
        ### TODO: IS THIS CORRECT???
        # ctrl_op_sin = obj_QET_sin.getCirc().to_gate().control(num_ctrl_qubits=1, ctrl_state="1") 

        ### TODO =============================================================================================
        op_sin = obj_QET_sin.getCirc().copy()        
        op_sin.append(Operator((0-1j) * np.identity(2 ** (1 + self.num_BE_qubits + self.num_qubits))),
                      [i for i in range(1 + self.num_BE_qubits + self.num_qubits-1, -1, -1)]) 
        ctrl_op_sin = op_sin.to_gate().control(num_ctrl_qubits=1, ctrl_state="1")        
        ### =============================================================================================


        
        main_circ.h(qreg_HS_ctrl)
        main_circ.append(ctrl_op_cos,
                        [qreg_HS_ctrl] + [qreg_QET_input[i] for i in range(1 + self.num_BE_qubits + self.num_qubits)])
        main_circ.append(ctrl_op_sin,
                        [qreg_HS_ctrl] + [qreg_QET_input[i] for i in range(1 + self.num_BE_qubits + self.num_qubits)])
        main_circ.h(qreg_HS_ctrl)
                
        self.HamSim_circ = main_circ

        return 
    

    def getCirc(self):
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")           
        return self.HamSim_circ

    
    def drawCirc(self,
                output: str = "text",
                decomp: bool = False):
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
            return None

        if decomp:
            print("The (decomposed) circuit for Hamiltonian Simulation by QET.")
            print(self.HamSim_circ.decompose().draw(output=output))
            print("The (decomposed) QET circuit for cos(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order, self.truncation_order, self.truncation_order, 2*self.truncation_order+1))
            print(self.obj_QET_cos.drawCirc(output=output, decomp=decomp))
            print("The (decomposed) QET circuit for sin(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order+1, self.truncation_order+1, self.truncation_order, 2*self.truncation_order+2))
            print(self.obj_QET_sin.drawCirc(output=output, decomp=decomp))
            
        else:
            print("The circuit for Hamiltonian Simulation by QET.")
            print(self.HamSim_circ.draw(output=output))
            print("The QET circuit for cos(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order, self.truncation_order, self.truncation_order, 2*self.truncation_order+1))
            print(self.obj_QET_cos.drawCirc(output=output, decomp=decomp))
            print("The QET circuit for sin(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order+1, self.truncation_order+1, self.truncation_order, 2*self.truncation_order+2))
            print(self.obj_QET_sin.drawCirc(output=output, decomp=decomp))
    

    
    def runHamiltonianSimulator(self) -> (Statevector, Statevector):
        
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")           
            return self.HamSim_circ

        if self.simulator_BasicAer is not True:
            
            if self.starting_state is None:
                state = Statevector.from_int(0, 2 ** (1 + 1 + self.num_BE_qubits + self.num_qubits))
            else:                               

                # starting_state = np.flip(self.starting_state)
                # state = Statevector.from_int(convert_binary_to_int(starting_state), 2 ** (1 + 1 + self.num_BE_qubits + self.num_qubits))
                # if not state.is_valid():
                #     # print("The starting state {} is not a valid quantum state. Returning None.".format(self.starting_state))
                #     raise ValueError("The starting state {} is not a valid quantum state.".format(self.starting_state))

                starting_state = Statevector(self.starting_state)
                if not starting_state.is_valid():
                    # print("The starting state {} is not a valid quantum state. Returning None.".format(self.starting_state))
                    raise ValueError("The starting state {} is not a valid quantum state.".format(self.starting_state))
                
                starting_state = starting_state.reverse_qargs().data
                state = Statevector(np.kron(starting_state, np.array([1., 0., 0., 0., 0., 0., 0., 0.])))
           

            # print("Running the Hamiltonian Simulation by QET circuit now...")
            state = state.evolve(self.HamSim_circ)
            # print("HamSim_QET circuit successfully run. The output state vector is")
            
            ## TODO: cannot draw
            # state.draw("latex")

        else:
            simulator = BasicAer.get_backend("statevector_simulator")
            # print("Running the Hamiltonian Simulation by QET circuit now...")
            state = Statevector(simulator.run(transpile(self.HamSim_circ, simulator)).result().get_statevector())
            # print("HamSim_QET circuit successfully run. The output state vector is")
            
            ## TODO: cannot draw
            # state.draw("latex")
            
        
        # print("Post-selection on the input register begins. If the desired measurement outcome (all 0's state) on the HamSim_ctrl, signal_ctrl, and BE registers is not obtained after 1000 tries, an PostMsmtError is raised.")
        # For post-selection, run the HamSim circuit for at most 1000 times; if the desired msmt outcome (all 0's state) cannot be obtained, raise a PostMsmtError(Exception)
        for _ in range(1000):
            
            msmt_result, post_msmt_state = state.measure([i for i in range(1 + 1 + self.num_BE_qubits)])
          
            if msmt_result != "0" * (1 + 1 + self.num_BE_qubits):
                # print("Round {}: msmt result = {}. Try again.".format(_+1, msmt_result))
                continue
            else:
                # print("Round {}: msmt result = {}. Post-selection on the input register is successful!".format(_+1, msmt_result))
                # print("The state vector after the measurement is")
                ## TODO: cannot draw
                # print(post_msmt_state.draw(output="latex"))
                # print(post_msmt_state)

                # Trace out the measured qubits
                density_matrix = partial_trace(post_msmt_state, [i for i in range(1 + 1 + self.num_BE_qubits)])
                # print("The density matrix after the partical trace has purity = {} and can be visualized below:".format(density_matrix.purity()))
                ## TODO: cannot draw
                # plot_state_city(density_matrix)
                # density_matrix.draw(output="city")
                
                postselected_state = density_matrix.to_statevector()
                # print("\n\n The final state vector, i.e. the state after time evolution t = {} under the input Hamiltonian is".format(self.evolution_time))
                ## TODO: cannot draw
                # print(postselected_state.draw(output="latex"))
                
                return postselected_state, postselected_state.reverse_qargs()

        raise PostMsmtError("Post-selection after 1000 rounds failed.")
        
        return None
