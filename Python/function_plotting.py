import random, math, cmath
from math import pi as Pi
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_power

class sequenceGenerator:
    def __init__(self,n: int=10) -> None:
        self.n=n

    def randomSeq(self) -> list:
        '''
            Return: a random sequence of (default) size 10. 
                    comprised of values in [0,2Pi)
        '''
        return [2 * Pi * random.uniform(0, 1) for i in range(self.n)]
    
    def randomPiSeq(self, max_denom: int=20, max_numer: int=40) -> list:
        '''
            Return: a random sequence of (default) size 10. 
                    comprised of values that are (nice) quotients of Pi
        '''
        seq=[] # Define the empty seq. list
        for i in range(self.n):
            # Generate a random demoninator and numerator
            denominator, numerator=random.randint(1, max_denom),\
                                    random.randint(1, max_numer)
            
            # Append these to the seq. list and scale by Pi
            seq.append((numerator/denominator) * Pi)

        return seq
    
    def fixedPiSeq(self, v: float=0.5) -> list:
        '''
            Return: a sequence of (default) size 10. 
                    comprised of values v * Pi 
            v: (default = 0.5) the scalar each Pi value is scaled by
        '''
        # Scale Pi by v
        # Create a list of n of such
        return [v * Pi for i in range(self.n)]

    def BB1(self,x):
        return [Pi/2, -x, 2*x,0,-2*x,x]
    
class functionBlockEncoding:
    def __init__(self) -> None:
        pass

    def Exp(self,x: float) -> float: 
        # Define Exp for ease
        return cmath.exp(1j * x)
    
    def xRotMatrix(self, theta: float) -> np.array:
        # Define cos(x/2) and sin(x/2) for ease
        c, s = np.cos(theta * 0.5), np.sin(theta * 0.5)

        return np.array([[c, -1j * s],\
                         [-1j * s, c]], dtype=complex)
    
    def zRotMatrix(self, phi: float) -> np.array:
        return np.array([[self.Exp(phi), 0],\
                         [0, self.Exp(-phi)]], dtype=complex)
    
    def conjugateIterateMatrixSeq(self, theta: float, phiList: list) -> np.array:
        '''
            Return: the final matrix of the sequential application of
                    the phase angles

            theta: the polar angle
            phiList: a list of phase angles, phi
        '''

        # Initialise the product matrix as the initial phase
        prod = self.zRotMatrix(phiList[0])

        # Iterate through the list from element [1,...]
        for angle in phiList[1:]:
            phaseRot = self.zRotMatrix(angle)
            intermediateMatrix = np.dot(self.xRotMatrix(theta), phaseRot)
            prod = np.dot(prod, intermediateMatrix)

        return prod
    
    def conjugateIterateMatrixSeqPlot(self, phiList: list, shots: int=500) -> plt.plot:
        '''
            Return: a plot of the upper left element of conjugateIterateMatrixSeq
                    runs from 0 -> 3Pi

            phiList: a list of phase angles, phi
            shots: number of linspace elements
        '''
        # For now we extra the Re([0][0]) component
        x = np.linspace(0, 3*Pi, 500)

        # Compute values for each x.
        y = [self.conjugateIterateMatrixSeq(xi, phiList)[0, 0].real for xi in x]

        # Plot the function
        plt.plot(x, y)

        # Set the tick values
        plt.xticks(np.arange(0, 7*Pi/2, step=(Pi/2)), ['0','π/2','π','3π/2','2π','5π/2','3π'])
        plt.yticks(np.arange(-1, 2, step=1), ['-1','0','1'])
        plt.show() 

    def conjugateIterateMatrixSeqPlotAltAbs(self, phiList: list, xmin: float=-Pi, xmax: float=Pi, shots: int=500) -> plt.plot:
        '''
            Return: an alternate plot of the upper left element of abs(conjugateIterateMatrixSeq)^2
                    runs (default) from -Pi -> Pi

            phiList: a list of phase angles, phi
            xmin, xmax: minimum and maximum ranges for the x-axis of the graph
            shots: number of linspace elements
        '''
        x = np.linspace(xmin, xmax, 500)

        # Compute values for each x.
        y = [abs(self.conjugateIterateMatrixSeq(xi, phiList)[0, 0])**2 for xi in x]

        # Plot the function
        plt.plot(x, y)

        # Set the tick values
        if xmin == -Pi and xmax == Pi:
            plt.xticks(np.arange(-Pi,2*Pi, step=(Pi)), ['-π','0','π'])

        plt.yticks(np.arange(0, 2, step=1), ['0','1'])
        plt.show() 

    def chebyshevPlot(self, degree: int, span: bool=False) -> plt.plot:
        '''
            Return: a plot of the nth degree Chebyshev polynomial
                    as calculated by a different iterate matrix

            dgeree: an int for the degree of Chebyshev
            span: (default = false) show all Chebyshev's up to degree set True
        '''
        if span == False:
            x = np.linspace(0, 1, 500)
            y = [matrix_power(np.array([[xi, 1j * math.sqrt(1 - xi**2)], [1j * math.sqrt(1 - xi**2), xi]]),degree)[0,0] for xi in x]
            # plt.plot(x, matrix_power(np.array([[x, 1j * math.sqrt(1 - x**2)], [1j * math.sqrt(1 - x**2), x]]),degree)[0,0])
            plt.plot(x,y)
            plt.show() 
        else:
            x = np.linspace(-1, 1, 500)
            for i in range(degree+1):
                plt.plot(x,\
                            [matrix_power(np.array([[xi, 1j * math.sqrt(1 - xi**2)], [1j * math.sqrt(1 - xi**2), xi]]),i)[0,0] for xi in x],\
                            label = "T_{}(x)".format(i))
                
            plt.legend()
            plt.show()

    def iterateMatrix(self, theta: float, phi: float) -> np.array:
        '''
            Return: the iterate matrix (2x2) for QSP

            theta: the polar angle
            phi: the azimuthal angle
        '''
        # Define cos(x/2) and sin(x/2) for ease
        c, s = np.cos(theta * 0.5), np.sin(theta * 0.5)

        # Return the iterate matrix
        return np.array([[c, -1j * self.Exp(-phi) * s],
                            [-1j * self.Exp(phi) * s, c]], dtype=complex)
    
    def iterateMatrixSeq(self,theta: float, phiList: list) -> np.array:
        '''
            Return: the final matrix of the sequential application of
                    the phase angles

            theta: the polar angle
            phiList: a list of phase angles, phi
        '''
        # Initialize the product matrix as a 2x2 identity matrix
        prod = np.array([[1, 0], [0, 1]], dtype=complex)     

        # Iterate through the input matrices and compute the product
        for angle in phiList:
            prod = np.dot(self.iterateMatrix(theta,angle), prod)

        return prod
    
    def iterateMatrixSeqPlot(self, phiList: list, shots: int=500) -> plt.plot:
        '''
            Return: a plot of the upper left element of iterateMatrixSeq
                    runs from 0 -> 3Pi

            phiList: a list of phase angles, phi
            shots: number of linspace elements
        '''
        # For now we extra the Re([0][0]) component
        x = np.linspace(0, 3*Pi, 500)

        # Compute values for each x.
        y = [self.iterateMatrixSeq(xi, phiList)[0, 0].real for xi in x]

        # Plot the function
        plt.plot(x, y)

        # Set the tick values
        plt.xticks(np.arange(0, 7*Pi/2, step=(Pi/2)), ['0','π/2','π','3π/2','2π','5π/2','3π'])
        plt.yticks(np.arange(-1, 2, step=1), ['-1','0','1'])
        plt.show()    

    def iterateMatrixSeqPlotAlt(self, phiList: list, shots: int=500) -> plt.plot:
        '''
            Return: an plot of an alternate sequence where theta |-> -2*np.arccos(x)

            phiList: a list of phase angles, phi
            shots: number of linspace elements
        '''
        # For now we extra the Re([0][0]) component
        x = np.linspace(-1, 1, 500)

        # Compute values for each x.
        y = [self.iterateMatrixSeq(-2*np.arccos(xi), phiList)[0, 0].real for xi in x]

        # Plot the function
        plt.plot(x, y)
        plt.show()  




