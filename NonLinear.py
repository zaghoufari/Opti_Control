# NonLinear

import numpy as np
import Base as sq
import Tracking as tr

class NonLinear(tr.Tracking):

    DEFAULT_RELATIVE_COST_DIFF_IN_PERCENT = 1
    DEFAULT_ITERATIONS_NUMBER = 100

    @classmethod
    def CreateExactGradients(cls, functions, gradientsA, gradientsB, lstQ, lstZ, lstR, lstU, 
                 lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop):
        return cls(functions, gradientsA, gradientsB, None, None, lstQ, lstZ, lstR, lstU, 
                 lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop)


    @classmethod
    def CreateNumericGradients(cls, functions, incM, incX, lstQ, lstZ, lstR, lstU, 
                 lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop):
        return cls(functions, None, None, incM, incX, lstQ, lstZ, lstR, lstU, 
                 lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop)


    @classmethod
    def CreateExactGradientsSimple(cls, functions, gradientsA, gradientsB, Q, Z, r, u, xInit, dt, N, dlgtConditionToStop):
        [lstQ, lstZ, lstR, lstU, lstMInitTrajectory, lstXInitTrajectory] = NonLinear.__createHelper(Q, Z, r, u, xInit, N)
        return cls(functions, gradientsA, gradientsB, None, None, lstQ, lstZ, lstR, lstU, 
                   lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop)


    @classmethod
    def CreateNumericGradientsSimple(cls, functions, incM, incX, Q, Z, r, u, xInit, dt, N, dlgtConditionToStop):
        [lstQ, lstZ, lstR, lstU, lstMInitTrajectory, lstXInitTrajectory] = NonLinear.__createHelper(Q, Z, r, u, xInit, N)
        return cls(functions, None, None, incM, incX, lstQ, lstZ, lstR, lstU, 
                   lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop)


    @classmethod
    def __createHelper(cls, Q, Z, r, u, xInit, N):
        lstQ = []
        lstZ = []
        lstR = []
        lstU = []
        lstMInitTrajectory = []
        lstXInitTrajectory = []

        for i in range(N):
            lstQ.append(Q)
            lstZ.append(Z)
            lstR.append(r)
            lstU.append(u)
            lstMInitTrajectory.append(np.zeros([np.matrix(u).shape[0], 1]))
            lstXInitTrajectory.append(np.zeros([np.matrix(r).shape[0], 1]))

        lstXInitTrajectory[0] = np.matrix(xInit)
        return [lstQ, lstZ, lstR, lstU, lstMInitTrajectory, lstXInitTrajectory]


    def __init__(self, functions, gradientsA, gradientsB, incM, incX, lstQ, lstZ, lstR, lstU, 
                 lstMInitTrajectory, lstXInitTrajectory, dt, dlgtConditionToStop):

        self.isExactGradient = True
        if gradientsA is None and gradientsB is None:
            self.isExactGradient = False

        def ConditionToStopDefault(currCost, prevCost, iteration):
            return (abs(currCost - prevCost) / prevCost) * 100 <= NonLinear.DEFAULT_RELATIVE_COST_DIFF_IN_PERCENT or \
                    iteration > NonLinear.DEFAULT_ITERATIONS_NUMBER

        super().__init__(None, None, lstQ, lstZ, lstR, lstU, lstXInitTrajectory[0], dt)

        self.dimensionM = np.matrix(lstU[0]).shape[0]
        self.dimensionX = np.matrix(lstR[0]).shape[0]

        self.Functions = functions

        self.GradientsForState = gradientsA
        self.GradientsForControl = gradientsB
        
        self.incM = incM
        self.incX = incX

        self.isGradientDelegate = self.GradientsForState is not None and self. GradientsForControl is not None
        
        self.dlgtConditionToStop = None
        if dlgtConditionToStop is None:
            self.dlgtConditionToStop = ConditionToStopDefault
        else:
            self.dlgtConditionToStop = dlgtConditionToStop

        self.control = lstMInitTrajectory
        self.state = lstXInitTrajectory

        self.xInit = self.state[0];

        self.RelativeCostDifferenceInPerCent = 100
        self.Cost = -1
        self.NumOfIterations = -1


    def RunOptimization(self):
        prevCost = 0.0
        currCost = 0.0
        while not self.dlgtConditionToStop(currCost, prevCost, self.iteration):
            self.RunInverseAndDirect()

            prevCost = currCost
            currCost = self.CalculateCost()

            self.iteration += 1

        self.RelativeCostDifferenceInPerCent = (abs(currCost - prevCost) / prevCost) * 100
        self.Cost = currCost;
        self.NumOfIterations = self.iteration - 1;
        return self;


    def GetDiscreteMatricesAtThisStep(self, k):
        A = np.zeros([self.dimensionX, self.dimensionX])
        for i in range(self.dimensionX):
            for j in range(self.dimensionX):
                if self.isGradientDelegate: 
                    A[i, j] = self.GradientsForState(i, j)(k, self.dt, self.control[k], self.state[k])
                else:
                    dx = np.zeros([self.dimensionX, 1])
                    dx[j] = self.incX[j]
                    A[i, j] = (self.Functions(i)(k, self.dt, self.control[k], self.state[k] + dx) - 
                               self.Functions(i)(k, self.dt, self.control[k], self.state[k])) / self.incX[j]
 
        B = np.zeros([self.dimensionX, self.dimensionM])
        for i in range(self.dimensionX):
            for j in range(self.dimensionM):
                if self.isGradientDelegate:
                    B[i, j] = self.GradientsForControl(i, j)(k, self.dt, self.control[k], self.state[k])
                else:
                    dm = np.zeros([self.dimensionM, 1])
                    dm[j] = self.incM[j]
                    B[i, j] = (self.Functions(i)(k, self.dt, self.control[k] + dm, self.state[k]) - 
                               self.Functions(i)(k, self.dt, self.control[k], self.state[k])) / self.incM[j]
 
        #F = Matrix.UnitMatrix(dimensionX) + A * dt
        #H = B * dt
        return self.ObtainDiscreteMatrices(A, B, self.dt)


    def GetDesirableStateAtThisStep(self, k):
        return self.DesirableState(k) - self.state[k]


    def GetDesirableControlAtThisStep(self, k):
        return self.DesirableControl(k) - self.control[k]
    

    def CalculateStateForNextStep(self, k, m, x):
        xNew = np.zeros([self.dimensionX, 1])
        for i in range(self.dimensionX):
            xNew[i] = self.Functions(i)(k, self.dt, m, x) * self.dt + x[i]

        return xNew


    def OutputExecutionDetails(self):
        numberOfSteps = self.NumOfIterations
        relativeCostDiff = self.RelativeCostDifferenceInPerCent
        if self.isExactGradient:
            prefix = "Exact" 
        else: 
            prefix = "Numerical"
        print(prefix + " gradients calculation")
        print("Num. of steps = " + str(numberOfSteps))
        if numberOfSteps == 1:
            print("Is feedback constant? " + str(self.IsConstFeedbackFor1stIteration))
            print("Is input    constant? " + str(self.IsConstInputFor1stIteration))
        else:
            print("Relative cost difference " +  str(float(relativeCostDiff)) + " per cent")
        return self

    
