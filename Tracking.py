# Tracking

import numpy as np
import Base as sq
import HelperMethods as hm 

class Tracking(sq.Base):
    
    def __init__(self, lstA, lstB, lstQ, lstZ, lstR, lstU, xInit, dt):
        super().__init__()
        self.__initTracking(lstQ, lstZ, lstR, lstU, xInit, dt)
        self.lstA = lstA
        self.lstB = lstB

        self.lstQ = lstQ
        self.lstZ = lstZ
        self.lstR = lstR
        self.lstU = lstU
          
        self.IsConstFeedbackFor1stIteration = len(hm.Distinct(self.lstQ)) == 1 and len(hm.Distinct(self.lstZ)) == 1
        self.IsConstInputFor1stIteration = self.IsConstFeedbackFor1stIteration and \
                                           len(hm.Distinct(self.lstR)) == 1 and len(hm.Distinct(self.lstU)) == 1


    def __initTracking(self, lstQ, lstZ, lstR, lstU, xInit, dt):
        self.InitParams(xInit, np.matrix(lstU[0]).shape[0], dt, np.array(lstR).shape[0])


    def TransferStateMatrix(self, k):
        return self.lstA[k]


    def TransferControlMatrix(self, k):
        return self.lstB[k]
    

    def WeightsStateMatrix(self, k):
        return self.lstQ[k]
    

    def WeightsControlMatrix(self, k):
        return self.lstZ[k]
    

    def DesirableState(self, k):  
        return self.lstR[k]


    def DesirableControl(self, k):  
        return self.lstU[k]
    

    def GetDiscreteMatricesAtThisStep(self, k):
        return self.ObtainDiscreteMatrices(self.lstA[k], self.lstB[k], self.dt)


    def GetDesirableStateAtThisStep(self, k):
        return self.lstR[k]
    

    def GetDesirableControlAtThisStep(self, k):
        return self.lstU[k]
    

    def GetWeightsStateMatrixAtThisStep(self, k):
        return self.lstQ[k]
    

    def GetWeightsControlMatrixAtThisStep(self, k):
        return self.lstZ[k]


    
    