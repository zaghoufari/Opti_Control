
import numpy as np

import Base as sq
import Tracking as tr
import NonLinear as nl
import HelperMethods as hm 

print("Sequential Quadratic Optimization begins\n");

# Linear ----------------------------------------------------------------------------------------------------

print("\nLinear Tracking\n")

M = 1.0

A =  np.matrix(
    [[0, 1],
     [0, 0]]
)

B = np.matrix(
    [[0],
     [1 / M]]
)

Q =  np.matrix(
    [[1, 0],
     [0, 3]]
)

Z = np.matrix(
    [[0]]
)

r = np.matrix(
    [[10],
     [0]]
)

u = np.matrix(
    [[0]]
)

xInit = np.matrix(
    [[0],
     [0]]
)

dt = 1
N = 41

dimensionX = 2
dimensionM = 1

sq.Base().Init(A, B, Q, Z, r, u, xInit, dt, N) \
         .RunOptimization() \
         .Output2Csv("Linear", "time, m0, x1, x2") #1\
         #1.Plot("Linear_py.png", 0.2)



# Van der Pol -----------------------------------------------------------------------------------------------
# Source:
#
# "NONLINEAR-QUADRATIC OPTIMAL CONTROL PROBLEM BY STATE-CONTROL PARAMETERIZATION"
#      by Hussein Jaddu and Milan Vlach
#      page 10, chapter 4.

# dx0/dt = x1
# dx1/dt = (mu) * x1 * (1 - x0^2) - x0 + m0
#     where mu = 0.01 - 4.0 In our case mu=1
# Cost Function = INTEGRAL[0 - 5](x0^2 + x1^2 + m0^2) - time is just between 0 and 5

print("\nVan der Pol")

dt = 0.1
N = 51

Q =  np.matrix(
    [[1.0, 0.0],
     [0.0, 1.0]]
)

Z = np.matrix(
    [[1.0]]
)

r = np.matrix(
    [[0.0],
     [0.0]]
)

u = np.matrix(
    [[0.0]]
)

xInit = np.matrix(
    [[1.0],
     [0.0]]
)

mu = 1.0

def functions(i):
    if i == 0:
        return lambda k, deltaT, m, x: x[1]
    if i == 1:
        return lambda k, deltaT, m, x: mu * (1 - np.square(x[0])) * x[1] - x[0] + m[0]

def gradientsA(i, j):
    if i == 0 and j == 0:
        return lambda k, deltaT, m, x: 0.0
    if i == 0 and j == 1:
        return lambda k, deltaT, m, x: 1.0
    if i == 1 and j == 0:
        return lambda k, deltaT, m, x: -(2 * mu * x[0] * x[1] + 1)
    if i == 1 and j == 1:
        return lambda k, deltaT, m, x: mu * (1 - np.square(x[0]))
    return None

def gradientsB(i, j):
    if (j != 0):
        return None
    
    if i == 0:
        return lambda k, deltaT, m, x: 0
    if i == 1:
        return lambda k, deltaT, m, x: 1
    return None

# Exact gradients calculation
nlnExact = nl.NonLinear.CreateExactGradientsSimple(functions, gradientsA, gradientsB, Q, Z, r, u, xInit, dt, N,
                                lambda currCost, prevCost, iteration: iteration > 17) \
                       .RunOptimization() \
                       .Output2Csv("VanDerPol_ExactGradients", "time, m0, x0, x1") \
                       .OutputExecutionDetails() #1\
                       #1.Plot("VanDerPol_exact_py.png")
                       


# Numeric gradients calculation
delta = 0.1

incM = []
for i in range(dimensionM):
    incM.append(delta)

incX = []
for i in range(dimensionX):
    incX.append(delta)

nlnNumeric = nl.NonLinear.CreateNumericGradientsSimple(functions, incM, incX, Q, Z, r, u, xInit, dt, N,
                                lambda currCost, prevCost, iteration: iteration > 17) \
                         .RunOptimization() \
                         .Output2Csv("VanDerPol_NumericGradients", "time, m0, x0, x1") \
                         .OutputExecutionDetails() #1\
                         #1.Plot("VanDerPol_num_py.png")

# -----------------------------------------------------------------------------------------------------------

print("Sequential Quadratic Optimization ends\n");


