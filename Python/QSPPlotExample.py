'''
    Let us cover some examples of using the function_plotting script
'''
import QSPPlotModule as QSPplot
from math import pi as Pi

# Initialise 4 examples
# One with each sequence type
# One with a v = 1/5
exampleOne = exampleTwo = exampleThree = exampleFour = exampleFive = QSPplot.sequenceGenerator()
exampleOneSeq,\
    exampleTwoSeq,\
    exampleThreeSeq,\
    exampleFourSeq,\
    exampleFiveSeq = exampleOne.randomSeq(),\
                        exampleTwo.randomPiSeq(),\
                        exampleThree.fixedPiSeq(),\
                        exampleFour.BB1(),\
                        exampleFive.fixedPiSeq(0.2)

for val in ['One', 'Two', 'Three', 'Four', 'Five']:
    print("Random Sequence: \n Size = {}\n Sequenece: {} \n\n".format(len(eval('example{}Seq'.format(val))), eval('example{}Seq'.format(val))))

'''
    Now we have one of each example, let's start plotting some graphs for them.

    Physically, the graphs represent the upper left element of the matrix where theta is a function.
        Moreover, we look at variations of <0|U(x,seq)|0>

    For each plot type we have an associated function that produces the final matrix of said sequence.

    For example,
'''

# Call the conjugate iterate matrix method
# Set theta = 0 and pass exampleOneSeq and also exampleFiveSeq
conjExampleOne, conjExampleTwo = QSPplot.sequencePlotting().conjugateIterateMatrixSeq(0, exampleOneSeq),\
                                    QSPplot.sequencePlotting().conjugateIterateMatrixSeq(0, exampleFiveSeq)
print(conjExampleOne)
print(conjExampleTwo)

'''
    If we wanted to see how the upper left element may vary for theta, we can plot this with the 
        conjugateIterateMatrixSeqPlot
'''

conjExampleOnePlot, conjExampleTwoPlot = QSPplot.sequencePlotting().conjugateIterateMatrixSeqPlot(exampleOneSeq),\
                                    QSPplot.sequencePlotting().conjugateIterateMatrixSeqPlot(exampleFiveSeq)
# print(conjExampleOnePlot)
# print(conjExampleTwoPlot)

'''
    As expected, these plot are fairly difficult to interpret but nevertheless represent what we expect.

    To further this point, the QSPPlotModule has a dedicated ChebyshevPlot that takes a degree and plots
        (up to) that degree. For example,
'''

# Plot all Chebyshev polynomials UP TO degree 6
chebyshevExample = QSPplot.sequencePlotting().chebyshevPlot(6, True, 'Chebyshev Plot')

print(chebyshevExample)

'''
    Given that we can generate this, we can check that the sequences and their respective functions replicate
        the Chebyshev's where appropriate.

    We did not define the Chebyshev sequence above since the sequence can be found via a sequence of all zeros!

    Let's try plot the iterate matrix using a sequence of 6 zeros
'''

chebyshevCheckExample = QSPplot.sequencePlotting().iterateMatrixSeqPlotAlt([0,0,0,0,0,0], 'Checker')

# We expect the 6th order Chebyshev
print(chebyshevCheckExample)

'''
    This worked as expected!

    Let us now print both the iterate and conjiterate matrix plots for the defined sequences.
'''

for val in ['One', 'Two', 'Three', 'Four', 'Five']:
    PlotTypeA, PlotTypeB = QSPplot.sequencePlotting().iterateMatrixSeqPlot(eval('example{}Seq'.format(val)), 'PlotTypeA'),\
                            QSPplot.sequencePlotting().conjugateIterateMatrixSeqPlot(eval('example{}Seq'.format(val)), 'PlotTypeB')
    print(PlotTypeA),
    print(PlotTypeB)