# SequentialQuadratic

import numpy as np
import HelperMethods as hm

class Base:

    # This variable should be set to "True" to allow test output of intermediate matrices
    isTestPrintRequired = False

    def __init__(self):
        self.N = 0
        self.dt = 0.0
        self.A = []
        self.B = []
        self.Q = []
        self.Z = []
        self.r = []
        self.u = []
        self.xInit = []

        self.dimensionM = 0
        self.dimensionX = 0

        self.iteration = 1
        self.lstInputAndFeedback = []

        self.state = []
        self.control = []
        self.IsConstInputFor1stIteration = False
        self.IsConstFeedbackFor1stIteration = False

    
    def Init(self, A, B, Q, Z, r, u, xInit, dt, N):
        self.N = N
        self.dt = dt
        self.A = A
        self.B = B
        self.Q = Q
        self.Z = Z
        self.r = r
        self.u = u
        self.xInit = xInit

        self.state = []
        self.control = []

        self.dimensionM = np.matrix(self.B).shape[1]
        self.dimensionX = np.matrix(self.A).shape[0]

        self.InitParams(xInit, self.dimensionM, dt, N)
        
        self.IsConstInputFor1stIteration = True
        self.IsConstFeedbackFor1stIteration = True
        self.ConstInputFor1stIteration = np.zeros(self.dimensionX)
        self.ConstFeedbackFor1stIteration = np.zeros([self.dimensionX, self.dimensionM])
       
        [self.dimensionX, self.dimensionM] = np.matrix(self.B).shape
        [self.F, self.H] = self.ObtainDiscreteMatrices(A, B, dt)       
        return self


    def InitParams(self, xInit, uDimension, dt, N):
        self.N = N
        self.dt = dt

        for k in range(N):
            self.control.append(np.zeros([uDimension, 1]))
            self.state.append(np.zeros(np.matrix(xInit).shape))

        self.state[0] = xInit


    def ObtainDiscreteMatrices(self, a, b, dt):
        eps = 1e-10
        imax = 100

        shapeA = np.matrix(a).shape
        shapeB = np.matrix(b).shape

        factor = 1
        ef = np.identity(shapeA[0])
        sf = np.identity(shapeA[0])
        eh = np.zeros(shapeB)
        sh = np.zeros(shapeB)
        
        efPrev = ef
        ehPrev = eh

        for i in range(1, imax + 1):
            efPrev = ef
            ehPrev = eh
            factor *= dt / i
            ef = np.matmul(ef, a) * factor
            sf += ef
            eh = np.matmul(efPrev, b) * factor
            sh += eh
            if self.IsStop(ef, efPrev, eps) and self.IsStop(eh, ehPrev, eps):
                break

        return [sf, sh]


    def IsStop(self, a, b, eps):
        d = a - b
        [firstDimension, secondDimension] = np.matrix(d).shape
        
        for i in range(firstDimension):
            for j in range(secondDimension):
                if abs(d[i, j]) > eps:
                    return False

        return True


    def RunInverseAndDirect(self):
        self.InverseRun()
        self.DirectRun()
  

    def InverseRun(self):
        P = np.zeros([self.dimensionX, self.dimensionX])
        v = np.zeros([self.dimensionX, 1])

        self.lstInputAndFeedback = []

        Base.TracePrint("Inverse Run ********************************************")

        for k in range(self.N - 1, -1, -1):
            try:
                r = self.GetDesirableStateAtThisStep(k)
                u = self.GetDesirableControlAtThisStep(k)
                Q = self.GetWeightsStateMatrixAtThisStep(k)
                Z = self.GetWeightsControlMatrixAtThisStep(k)

                [F, H] = self.GetDiscreteMatricesAtThisStep(k)

                Base.TracePrint("F = ", F, k)
                Base.TracePrint("H = ", H, k)
               
                QP = Q + P
                Base.TracePrint("Q + P = ", QP, k)

                W_1 = np.linalg.inv(H.T * QP * H + Z)
                Base.TracePrint("W_1 = ", W_1, k)

                g = Q * r + v
                inputVector = W_1 * (H.T * g + Z * u)
                matrixFeedback = W_1 * H.T * QP * F

                Base.TracePrint("inputVector = ", inputVector, k)      
                Base.TracePrint("matrixFeedback = ", matrixFeedback, k)

                self.lstInputAndFeedback.append(InputAndFeedback(inputVector, matrixFeedback))

                if k <= 0:
                    continue

                E = F - H * matrixFeedback
                Base.TracePrint("E = ", E, k)      

                P = E.T * QP * E + matrixFeedback.T * Z * matrixFeedback

                Base.TracePrint("E.T * QP * E = ", E.T * QP * E, k)      
                Base.TracePrint("matrixFeedback.T * Z * matrixFeedback = ", matrixFeedback.T * Z * matrixFeedback, k)      

                v = ((g.T - (H * inputVector).T * QP) * E + (inputVector - u).T * Z * matrixFeedback).T

                Base.TracePrint("P = ", P, k)
                Base.TracePrint("v = ", v, k)
            except ValueError:
                print("Exception in InverseRun, ", k, ValueError)


    def DirectRun(self):
        Base.TracePrint("Direct Run ******************************************")
        x = self.xInit          
        for k in range(self.N):
            Base.TracePrint("...................................................")

            self.SetStateAtThisStep(k, x)
            
            s = self.N - k - 1
            c = self.lstInputAndFeedback[s].Input
            L = self.lstInputAndFeedback[s].Feedback
            if self.iteration == 1:
                s = self.N - 2
                    
                if self.IsConstInputFor1stIteration:
                    self.ConstInputFor1stIteration = c = self.lstInputAndFeedback[s].Input

                if self.IsConstFeedbackFor1stIteration:
                    self.ConstFeedbackFor1stIteration = L = self.lstInputAndFeedback[s].Feedback
                 
            m = c - L * x + self.GetCurrentControlAtThisStep(k)

            Base.TracePrint("x = ", x, k)
            Base.TracePrint("c = ", c, k)
            Base.TracePrint("L = ", L, k)
            Base.TracePrint("self.GetCurrentControlAtThisStep(k) = ", self.GetCurrentControlAtThisStep(k), k)
                       
            self.SetControlAtThisStep(k, m)
            x = self.CalculateStateForNextStep(k, m, x)
 
            Base.TracePrint("m = ", m, k)
            Base.TracePrint("x = ", x, k)


    def RunOptimization(self):
        self.RunInverseAndDirect()
        self.Cost = self.CalculateCost()
        return self
        

    def CalculateCost(self):
        cost = 0
        for k in range(0, self.N - 1): 
            r = self.GetDesirableStateAtThisStep(k + 1)
            u = self.GetDesirableControlAtThisStep(k)
            Q = self.GetWeightsStateMatrixAtThisStep(k)
            Z = self.GetWeightsControlMatrixAtThisStep(k)
            x = self.GetCurrentStateAtThisStep(k + 1)
            m = self.GetCurrentControlAtThisStep(k)
                
            diffX = r - x
            diffM = u - m
            costAtThisStep = diffX.T * Q * diffX + diffM.T * Z * diffM

            cost += costAtThisStep
        return cost
        

    def GetDiscreteMatricesAtThisStep(self, k):
        return [self.F, self.H]
                    

    def GetDesirableStateAtThisStep(self, k):
        return self.r
        

    def GetDesirableControlAtThisStep(self, k):
        return self.u
        

    def GetWeightsStateMatrixAtThisStep(self, k):
        return self.Q
        

    def GetWeightsControlMatrixAtThisStep(self, k):
        return self.Z

        
    def GetCurrentStateAtThisStep(self, k):
        return self.state[k]
        

    def GetCurrentControlAtThisStep(self, k):
        return self.control[k]
        

    def SetControlAtThisStep(self, k, m):
        self.control[k] = m
        

    def SetStateAtThisStep(self, k, x):      
        self.state[k] = x
        

    def CalculateStateForNextStep(self, k, m, x):
        [F, H] = self.GetDiscreteMatricesAtThisStep(k)
        return F * x + H * m


    def Output2Csv(self, path, headline):
        filePath = path + ".csv"       
        with open(filePath, "w") as f:
            f.write(headline + "\n")
            for k in range(self.N):
                f.write(str(k * self.dt) + ",")

                vals = []

                for i in range(self.dimensionM):
                    vals.append(self.GetCurrentControlAtThisStep(k)[i])

                for i in range(self.dimensionX):
                    vals.append(self.GetCurrentStateAtThisStep(k)[i])

                hm.PrintLineCsv(f, vals)
                f.write("\n")
        return self


    def Plot(self, path = "", size = 1, title = "", xlabel = "", ylabel = ""):
        import matplotlib.pyplot as plt
        from scipy.interpolate import make_interp_spline, BSpline
        
        length = len(self.state)
        if size > 0 and size < 1:
            length = int(length * size)

        trange = self.dt * length
        tnew = np.linspace(0, trange, num=length*10)
        
        # Data for plotting
        t = np.arange(0.0, trange, self.dt)

        plots = []
        for i in range(self.dimensionM + self.dimensionX):
            s = []
            for k in range(length):
                if i < self.dimensionM:
                    s.append(float(self.control[k][i]))  
                else:
                    s.append(float(self.state[k][i - self.dimensionM])) 

            spl = make_interp_spline(t, s, k=3) # BSpline object
            plots.append(plt.plot(tnew, spl(tnew)))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()

        curves = []
        curveNames = []
        for i in range(len(plots)):
            curves.append(plots[i][0])
            if i < self.dimensionM:
                curveNames.append("m" + str(i))
            else:
                curveNames.append("x" + str(i - self.dimensionM)) 

        plt.legend(curves,              # plot items
                   curveNames,          # titles
                   frameon=True,        # legend border
                   framealpha=1,        # transparency of border
                   ncol=1,              # num columns
                   shadow=True,         # shadow on
                   borderpad=1,         # thickness of border
                   title='')            # title

        if len(path) > 0:
            plt.savefig(path)

        plt.show()
        return self

    
    def TracePrint(title, value = None, k = 0):
        if Base.isTestPrintRequired:
            print("k = ", k)
            print(title)
            if value is not None:
                print(value)
            print("\n")



class InputAndFeedback:
    def __init__(self, input, feedback):
        self.Input = input
        self.Feedback = feedback
 

