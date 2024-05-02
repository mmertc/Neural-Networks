import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.special import softmax as softmaxx
import h5py
from sklearn.utils import shuffle
import cupy as cp


class RNN_Network:

    def __init__(self, learnConst, hidLayerN, icputLength) -> None:
        self.learnConst_ = learnConst
        self.hidLayerN_ = hidLayerN
        self.icputLength_ = icputLength


        self.WHO_time_ = 0
        self.WHH_time_ = 0
        self.WIH_time_ = 0
        self.tensorProduct_time_ = 0
        self.FlatToTensor3_timer_ = 0
        self.WHH_inside_timer_ = 0

        """ self.samplesAmount_ = samplesAmount
        self.timeseriesLenght_ = timeseriesLenght
        self.icputLength_ = icputLength """

        """ self.samplesAmount_ = X.shape[0]
        self.timeseriesLenght_ = X.shape[1]
        self.icputLength_ = X.shape[2] """

    def tanh(x):
        return cp.tanh(x)
    
    def Dtanh(x):
        return 1 - cp.square(RNN_Network.tanh(x)) 
    
    def softmax(x):
        x = cp.asnumpy(x)
        return cp.asarray(softmaxx(x, axis=1))
    
    def Dsoftmax(x):
        return RNN_Network.softmax(x) * (1 - RNN_Network.softmax(x))
    
    def extendIcput(X):
        rows = X.shape[0]
        ones = cp.ones((rows, 1))
        return cp.concatenate((X, -1 * ones), axis=1)
    
    def forwardTensor(self, W, X):
        return cp.transpose(cp.matmul(W, cp.transpose(X, axes=(0, 2, 1))), axes=(0, 2, 1))
    
    def forward(self, W, X):
        return cp.transpose(W @ cp.transpose(X))
    
    def activation(self, func, Z):
        return func(Z)
    
    def applyLayer(self, func, W, X):
        Z = self.forward(W, X)
        Y = self.activation(func, Z)
        return Y
    
    def calculateStates(self, func, WIH, WHH, S, X): #consider running in parallel

        S_act = S
        S_unact = S
        S_unact[:, 1, :] = cp.transpose(cp.matmul(cp.asarray(WHH), cp.asarray(cp.transpose(S_unact[:, 1, :]))))

        for i in range(S_act.shape[0]):
            for j in range(S_act.shape[1]):
                k = S_act[i, j, :]
                if not (j == S_act.shape[1]-1):
                    k_forward = WHH @ k
                    v = k_forward + WIH @ X[i, j, :]
                    v_act = func(v) 
                                            
                    S_act[i, j+1, :] = v_act
                    S_unact[i, j+1, :] = v

        return S_act, S_unact
    
    
    def initWeights(self):
        self.WIH_ = (0.01 - -0.01) * cp.random.rand(self.hidLayerN_, self.icputLength_+1) + -0.01
        self.WIH_[-1,:] = 0
        self.WHH_ = (0.01 - -0.01) * cp.random.rand(self.hidLayerN_, self.hidLayerN_) + -0.01
        self.WHH_[-1,:] = 0
        self.WHO_ = (0.01 - -0.01) * cp.random.rand(6, self.hidLayerN_ +1) + -0.01
        self.WHO_[-1,:] = 0

    def initStates(self):
        self.S_ = cp.empty((self.samplesAmount_, self.timeseriesLenght_, self.hidLayerN_))
        self.S_[:, 0, :] = (0.01 - -0.01) * cp.random.rand(self.samplesAmount_, self.hidLayerN_) + -0.01


    def crossEntropyLoss(Y, labels):
        epsilon=1e-5
        return cp.sum((labels * cp.clip(cp.log(Y+epsilon), -1000, 1000)), axis=1) * -1
    
    def crossEntropyLoss_np(Y, labels):
        epsilon=1e-5
        return np.sum((labels * np.clip(np.log(Y+epsilon), -1000, 1000)), axis=1) * -1


    def forwardNetworkAndSave(self, funcHidden, funcOut, funcCost, WIH, WHH, WHO, X, D):
        Xe = self.extend(X)

        self.FIH_ = self.forwardTensor(WIH, Xe)
        self.SAct_, self.S_unAct_ = self.calculateStates(funcHidden, WIH, WHH, self.S_, Xe)

        self.V_ = self.FIH_ + self.S_unAct_

        self.O_ = self.forward(WHO, self.extend(self.SAct_[:, -1, :]))
        #self.O_ = cp.expand_dims(self.O_, axis=1)

        self.Y_ = self.activation(funcOut, self.O_)

        self.E_ = funcCost(self.Y_, D)


    """ def calculate_DhDhp(self, dForwards, WHH):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], WHH.shape[1]))
        for i in range(dForwards.shape[0]):
            for j in range(dForwards.shape[1]):            
                product = cp.diag(dForwards[i, dForwards.shape[1]-1, :]) @ WHH
                for k in range(1, j):
                    product = product @ cp.diag(dForwards[i, dForwards.shape[1]-1-k, :]) @ WHH
                result[i, j, :, :] = product

        return result """
    

    def calculate_DhDhp(self, dForwards, WHH):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], WHH.shape[1]))

        for i in range(dForwards.shape[0]):
            diag_matrices = cp.diag(dForwards[i, dForwards.shape[1]-1, :])
            product = cp.matmul(cp.asarray(diag_matrices), cp.asarray(WHH))

            for j in range(dForwards.shape[1]):
                for k in range(1, j):
                    diag_matrices = cp.matmul(cp.asarray(cp.diag(dForwards[i, dForwards.shape[1]-1-k, :])), cp.asarray(WHH))
                    product = cp.matmul(cp.asarray(product), cp.asarray(diag_matrices))
                    #product = product * (10/cp.linalg.norm(product))

            #print(cp.max(product), cp.min(product), cp.linalg.norm(product))
            result[i, j, :, :] = product
            

        return result



    """  def calculate_DhDWHH(self, dForwards, S_Act):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], S_Act.shape[2]))
        for i in range(dForwards.shape[0]):
            for j in range(dForwards.shape[1]):
                result[i, j, :, :] = cp.diag(dForwards[i, dForwards.shape[1]-1-j, :]) @ S_Act[i, S_Act.shape[1]-1-(j+1), :]

        return result 

    def calculate_DhDWIH(self, dForwards, Xe):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], Xe.shape[2]))
        for i in range(dForwards.shape[0]):
            for j in range(dForwards.shape[1]):
                result[i, j, :, :] = cp.diag(dForwards[i, dForwards.shape[1]-1-j, :]) @ Xe[i, Xe.shape[1]-1-(j+1), :]

        return result """
    
    def calculate_DhDWHH(self, dForwards, S_Act):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], dForwards.shape[2], S_Act.shape[2]))
        for i in range(dForwards.shape[0]):
            for j in range(dForwards.shape[1]):
                DhiDv = cp.diag(dForwards[i, dForwards.shape[1]-1-j, :])
                DvDWHH_flat = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], S_Act.shape[2]))
                for k in range(DvDWHH_flat.shape[2]):
                    DvDWHH_flat[i, j, k, :] = S_Act[i, j, :] 
                
                DhiDWHH = self.matrix_tensor3_Product_3(DhiDv, DvDWHH_flat[i, j, :, :])
                     
                
                
                result[i, j, :, :, :] = DhiDWHH

        return result
    


    
    """ def calculate_DhDWHH_optimized(self, dForwards, S_Act):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], dForwards.shape[2], S_Act.shape[2]))

        for i in range(dForwards.shape[0]):
            for j in range(dForwards.shape[1]):
                DhiDv = cp.diag(dForwards[i, dForwards.shape[1]-1-j, :])
                DvDWHH = cp.tile(S_Act[i, j, cp.newaxis, :], (dForwards.shape[2], 1))
                
                DhiDWHH = self.matrix_tensor3_Product_3(DhiDv, DvDWHH)
                result[i, j, :, :, :] = DhiDWHH

        return result """





    
    def matrix_tensor3_Product_3(self, A, T):
        result = A[:, :, cp.newaxis] * T
        return result
    


    
    def calculate_DhDWIH(self, dForwards, Xe):
        result = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], dForwards.shape[2], Xe.shape[2]))
        for i in range(dForwards.shape[0]):
            for j in range(dForwards.shape[1]):
                DhiDv = cp.diag(dForwards[i, dForwards.shape[1]-1-j, :])
                DvDWIH_flat = cp.empty((dForwards.shape[0], dForwards.shape[1], dForwards.shape[2], Xe.shape[2]))
                for k in range(DvDWIH_flat.shape[2]):
                    DvDWIH_flat[i, j, k, :] = Xe[i, j, :]

                #DhiDWIH = matrix_tensor3_Product_optimized(DhiDv, DvDWIH[i, j, :, :, :])

                #DvDWIH = self.matrixFlatToTensor3(DvDWIH_flat) 
                #DhiDWIH = self.matrix_tensor3_Product(DhiDv, DvDWIH[i, j, :, :, :])   

                DhiDWIH = self.matrix_tensor3_Product_3(DhiDv, DvDWIH_flat[i, j, :, :])     
                
                
                result[i, j, :, :, :] = DhiDWIH

        return result

    """ def matrix_tensor3_Product(self, A, T):
        start = timer()

        result = cp.empty((A.shape[0], T.shape[1], T.shape[2]))

        for i in range(A.shape[0]):
            sum = cp.zeros((T.shape[1], T.shape[2]))
            for j in range(A.shape[1]):
                sum = sum + A[i, j] * T[j, :, :]
            result[i, :, :] = sum
        
        end = timer()

        self.tensorProduct_time_ = self.tensorProduct_time_ + end-start

        return result """


    """ def matrix_tensor3_Product(self, A, T):
        start = timer()

        result = cp.einsum('ij,jkl->ikl', A, T)

        end = timer()

        self.tensorProduct_time_ += end - start

        return result """
    

    import cupy as cp

    def matrix_tensor3_Product(self, A, T):
        start = timer()

        A_gpu = cp.asarray(A)
        T_gpu = cp.asarray(T)

        result_gpu = cp.einsum('ij,jkl->ikl', A_gpu, T_gpu)

        result = result_gpu  

        end = timer()

        self.tensorProduct_time_ += end - start

        return result

    

    def matrix_tensor3_Product2(self, A, T):
        result = A.flatten()[:, cp.newaxis] * T
        return result

    

    def matrixFlatToTensor3(self, A):
        start = timer()

        result = cp.empty((A.shape[0], A.shape[1], A.shape[2], A.shape[2], A.shape[3]))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    result[i, j, k, :, :] = cp.zeros((A.shape[2], A.shape[3]))                
                    result[i, j, k, k, :] = A[i, j, k, :]

        end=timer()
        self.FlatToTensor3_timer_ = self.FlatToTensor3_timer_ + end-start
        return result
    

    """ def matrix_tensor3_Product_optimized(self, A, T):
        #start = timer()

        result = cp.einsum('ijkl,jklm->ikm', A, T)

        end = timer()

        #self.tensorProduct_time_ += end - start

        return result """






    """ def matrixFlatToTensor3(self, A):
        start = timer()

        result = cp.einsum('ijklm,ijm->ijklo', cp.eye(A.shape[2])[:, None, None,:], A)

        end = timer()
        self.FlatToTensor3_timer_ += end - start

        return result """



    def calculateHOGradients(self, funcOutD, S_Act_extended, WHO):
        outForwards = WHO @ S_Act_extended[:, -1, :]
        outActivatedD = funcOutD(outForwards)

        result = cp.empty((outActivatedD.shape[0], outActivatedD.shape[1], S_Act_extended.shape[2]))

        for i in range(outActivatedD.shape[0]):
            result[i,:,:] = cp.diag(outActivatedD[i, :]) @ S_Act_extended[i, -1, :]

        return result
    
    def calculate_DeDy(self, Y, D):
        epsilon=1e-5
        dY = -1 * cp.clip(cp.reciprocal(Y+epsilon), -10000, 10000)       
        result = dY * D

        return result
    
    def calculate_DyDv(self, Y):
        result = cp.empty((Y.shape[0], Y.shape[1], Y.shape[1]))
        for k in range(Y.shape[0]):
            for i in range(Y.shape[1]):
                for j in range(Y.shape[1]):
                    if i == j:
                        result[k, i, i] = Y[k, i] * (1 - Y[k, i])
                    else:
                        result[k, i, j] = -1 * Y[k, i] * Y[k, j]

        return result
    

    def calculate_DvDh(self, WHO):
        return WHO[:,:-1]
    


    def calculate_DvDWHO(self, WHO_shape, h):
        result = cp.empty((h.shape[0], WHO_shape[0], WHO_shape[0], WHO_shape[1]))

        for k in range(h.shape[0]):
            for i in range(WHO_shape[0]):
                zeros = cp.zeros((WHO_shape[0], WHO_shape[1]))
                zeros[i, :] = h[k, :]

                result[k, i,  :, :] = zeros

        return result
    

    def calculate_DvDWHO_2(self, WHO_shape, h):
        result = cp.empty((h.shape[0], WHO_shape[0], WHO_shape[1]))

        for k in range(h.shape[0]):
            result[k, :, :] = cp.vstack([h[k, :]] * WHO_shape[0])

        return result
                





    def calculate_DWHO(self, Y, D, WHO, he_last):
        DeDy = self.calculate_DeDy(Y, D)
        DeDy = cp.expand_dims(DeDy, axis=1)

        DyDv = self.calculate_DyDv(Y)
        DvDWHO = self.calculate_DvDWHO_2(WHO.shape, he_last)

        DeDv = DeDy @ DyDv

        sum = cp.zeros((DeDv.shape[0], DeDv.shape[1], DvDWHO.shape[-1]))
        for i in range(DeDv.shape[0]):
            sum = sum + self.matrix_tensor3_Product2(DeDv[i, :, :],DvDWHO[i, :, :])

        DWHO = cp.sum(sum, axis=0)   
        dWHO =  DWHO/cp.linalg.norm(DWHO)
        return dWHO


    """ def calculate_DWHH(self, X, Y, D, WHO, WHH, dForwards, S_act):
        DeDy = self.calculate_DeDy(Y, D)
        DyDv = self.calculate_DyDv(Y)
        DvDh = self.calculate_DvDh(WHO)
        DhDhp = self.calculate_DhDhp(dForwards, WHH)
        DhDWHH = self.calculate_DhDWHH(dForwards, S_act)

        DWHH = cp.empty((X.shape[0], X.shape[1], DeDy.shape[0], DhDWHH.shape[-1]))

        for k in range(X.shape[0]):
            sum = cp.zeros((DhDhp.shape[2], DhDhp.shape[3]))
            for i in range(X.shape[1]):

                product = cp.eye(DhDhp.shape[3])
                for j in range(i):
                    product = DhDhp[k, j, :, :] @ product
                product = product @ DhDWHH
                sum = sum + product
            DWHH[k, :, :] = DeDy @ DyDv @ DvDh @ sum
        
        dWHH = cp.sum(DWHH, axis=1)/(DWHH.shape[1] * X.shape[1])
        return dWHH """
    

    def calculate_DWHH(self, X, Y, D, WHO, WHH, dForwards, S_act):
        

        DeDy = cp.asarray(self.calculate_DeDy(Y, D))
        DeDy = cp.expand_dims(DeDy, axis=1)

        DyDv = cp.asarray(self.calculate_DyDv(Y))

        DvDh = cp.asarray(self.calculate_DvDh(WHO))

        start = timer()
        
        DhDhp = cp.asarray(self.calculate_DhDhp(dForwards, WHH))
        DhDWHH = cp.asarray(self.calculate_DhDWHH(dForwards, S_act))

        end= timer()
        self.WHH_inside_timer_ = self.WHH_inside_timer_ + end-start

        DWHH = cp.empty((X.shape[0], WHH.shape[1], S_act.shape[2]))

        

        for k in range(X.shape[0]):
            sum = cp.zeros((DhDhp.shape[2], DhDhp.shape[3], DhDWHH.shape[-1]))
            for i in range(X.shape[1]):

                product = cp.eye(DhDhp.shape[2])
                if i == 0:
                    #product = cp.zeros((DhDhp.shape[2], DhDhp.shape[3]))
                    product = cp.eye(DhDhp.shape[2])
                else:
                    product = DhDhp[k, DhDhp.shape[1]-i-1, :, :] @ product
                
                #for j in range(i):
                    #product = DhDhp[k, DhDhp.shape[1]-j-1, :, :] @ product
                #product = self.matrix_tensor3_Product(product, DhDWHH[k, i, :, :, :])

                product = cp.einsum('ij,jkl->ikl', product, DhDWHH[k, i, :, :, :])
                

                sum = sum + product
            DeDh = DeDy @ DyDv @ DvDh

            DeDWHH = cp.einsum('ij,jkl->ikl',DeDh[k, :, :], sum)
            
            

        DWHH = DeDWHH
        DWHH = cp.sum(DWHH, axis=0)

        dWHH = DWHH/cp.linalg.norm(DWHH)
        return dWHH
    

    """ def calculate_DWHH(self, X, Y, D, WHO, WHH, dForwards, S_act):
        DeDy = cp.expand_dims(self.calculate_DeDy(Y, D), axis=1)
        DyDv = self.calculate_DyDv(Y)
        DvDh = self.calculate_DvDh(WHO)
        DhDhp = self.calculate_DhDhp(dForwards, WHH)
        DhDWHH = self.calculate_DhDWHH(dForwards, S_act)

        DhDhp_cumprod = cp.cumprod(cp.rollaxis(DhDhp[:, ::-1, :, :], 1, 0), axis=0)[::-1, :, :, :]
        DhDhp_cumprod = cp.rollaxis(DhDhp_cumprod, 1, 0)

        product = cp.einsum('ijklm,ijmno->inokl', DhDhp_cumprod, DhDWHH)

        DeDh = cp.einsum('ijk,ijl,ijl->ijk', DeDy, cp.expand_dims(DyDv, axis=0), DvDh)

        DWHH = cp.einsum('ijk,ijklm->ilm', DeDh, product)

        dWHH = DWHH / cp.linalg.norm(DWHH)

        return dWHH """





    

    def calculate_DWIH(self, X, Y, D, WHO, WHH, dForwards, Xe):
        DeDy = self.calculate_DeDy(Y, D)
        DyDv = self.calculate_DyDv(Y)
        DvDh = self.calculate_DvDh(WHO)
        DhDhp = self.calculate_DhDhp(dForwards, WHH)
        DhDWIH = self.calculate_DhDWIH(dForwards, Xe)

        DWIH = cp.empty((X.shape[0], WHH.shape[1], Xe.shape[2]))

        for k in range(X.shape[0]):
            sum = cp.zeros((DhDhp.shape[2], DhDhp.shape[3], DhDWIH.shape[-1]))
            for i in range(X.shape[1]):

                """ product = cp.eye(DhDhp.shape[3])
                for j in range(i):
                    product = DhDhp[k, DhDhp.shape[1]-j-1, :, :] @ product
                product = self.matrix_tensor3_Product(product, DhDWIH[k, i, :, :, :]) """

                product = cp.eye(DhDhp.shape[2])
                if i == 0:
                    #product = cp.zeros((DhDhp.shape[2], DhDhp.shape[3]))
                    product = cp.eye(DhDhp.shape[2])
                else:
                    product = cp.matmul(cp.asarray(DhDhp[k, DhDhp.shape[1]-i-1, :, :]), cp.asarray(product))

                product = self.matrix_tensor3_Product(product, DhDWIH[k, i, :, :, :])
              
                sum = sum + product
            DeDh = DeDy @ DyDv @ DvDh   
            DeDWIH = self.matrix_tensor3_Product(DeDh[k, :, :], sum)
           
            
        DWIH = DeDWIH
        DWIH = cp.sum(DWIH, axis=0)

        dWIH = DWIH/cp.linalg.norm(DWIH)
        return dWIH

    def extend(self, X):
        X = cp.asarray(X)
        mOnes = cp.expand_dims(cp.ones((X.shape[:-1])) * -1, axis=len(X.shape)-1)
        Xe = cp.concatenate((X, mOnes), axis=len(X.shape)-1)
        return Xe
    
    def extend_np(self, X):
        X = cp.asnumpy(X)
        mOnes = np.expand_dims(np.ones((X.shape[:-1])) * -1, axis=len(X.shape)-1)
        Xe = np.concatenate((X, mOnes), axis=len(X.shape)-1)
        return Xe
    


    def switch_states(self):
        self.S_[:, 0, :] = self.S_[:, -1, :]


    def forget_states(self):
        self.S_[:, 0, :] = (0.01 - -0.01) * cp.random.rand(self.samplesAmount_, self.hidLayerN_) + -0.01


    def predict(self, X):

        Xe = self.extend(X)

        #FIH = self.forwardTensor(self.WIH_, Xe)

        S = cp.empty((X.shape[0], X.shape[1], self.hidLayerN_))
        S[:, 0, :] = (0.01 - -0.01) * cp.random.rand(X.shape[0], self.hidLayerN_) + -0.01


        SAct, S_unAct= self.calculateStates(RNN_Network.tanh, self.WIH_, self.WHH_, S, Xe)

        #V = FIH + S_unAct

        O = self.forward(self.WHO_, self.extend(SAct[:, -1, :]))
        Y = self.activation(RNN_Network.softmax, O)

        preds = cp.asarray(np.argmax(cp.asnumpy(Y), axis=1))
        return preds
    
    
    def accuracy(self, preds, labels):
        acc = cp.mean(preds == cp.argmax(labels, axis=1))
        return acc
    
    def accuracy_np(self, preds, labels):
        acc = np.mean(preds == np.argmax(labels, axis=1))
        return acc


    def print_timers(self):
        print("WHO Timer: ", self.WHO_time_, ", WHH Timer: ", self.WHH_time_, ", WIH Timer: ", self.WIH_time_, ", Total Train Time: ", self.train_time_, ", Tensor Product Time: ", self.tensorProduct_time_, " Flat to Tensor3 Timer: ", self.FlatToTensor3_timer_, ", WHH Inside Timer: ", self.WHH_inside_timer_)


    def run_BPTT_(self, batch, D):
        self.forwardNetworkAndSave(RNN_Network.tanh, RNN_Network.softmax, RNN_Network.crossEntropyLoss, self.WIH_, self.WHH_, self.WHO_, batch, D)
        
        he_lasts = self.extend(self.SAct_[:, -1, :])
        batch_extended = self.extend(batch)
        dForwards = RNN_Network.Dtanh(self.V_)
        dForwards_extended = self.extend(dForwards)


        start = timer()
        DWHO = self.calculate_DWHO(self.Y_, D, self.WHO_, he_lasts)
        end = timer()
        self.WHO_time_ = self.WHO_time_ + end-start
        #print("WHO_time: ", end-start)

        start = timer()
        DWHH = self.calculate_DWHH(batch, self.Y_, D, self.WHO_, self.WHH_, dForwards, self.SAct_)
        end = timer()
        self.WHH_time_ = self.WHH_time_ + end-start
        #print("WHH_time: ", end-start)

        start = timer()
        DWIH = self.calculate_DWIH(batch, self.Y_, D, self.WHO_, self.WHH_, dForwards, batch_extended)
        end = timer()
        self.WIH_time_ = self.WIH_time_ + end-start
        #print("WIH_time: ", end-start)


        self.WHO_ = self.WHO_ + (-1 * self.learnConst_ * DWHO)
        self.WHH_ = self.WHH_ + (-1 * self.learnConst_ * DWHH)
        self.WIH_ = self.WIH_ + (-1 * self.learnConst_ * DWIH)



    def run_BPTT(self, X, D, batch_size, time_length, epochs):


        start = timer()

        n_X = X.shape[0] / batch_size
        n_T = X.shape[1] / time_length


        batches_in_samples = np.array_split(X, n_X, axis=0)
        desireds = np.array_split(D, n_X, axis=0)

        for n in range(epochs):
            print("Epoch number: ", n+1)

            for i, j in zip(batches_in_samples, desireds):
                batches_in_both = np.array_split(i, n_T, axis=1)
                
                for k in batches_in_both:


                    self.samplesAmount_ = k.shape[0]               
                    self.timeseriesLenght_ = k.shape[1]

                    self.initStates()
                
                    #start = timer()
                    self.run_BPTT_(cp.asarray(k), cp.asarray(j))
                    #end = timer()
                    #print("Time for batch: ", end-start)


                    self.switch_states()
                self.forget_states()
            
            self.predictAndLossAndAcc(X, D)

        end = timer()
        self.train_time_ = end-start


    def predictAndLossAndAcc(self, X, D):

        X =cp.asarray(X)
        D=cp.asarray(D)

        Xe = self.extend(X)

        S = cp.empty((X.shape[0], X.shape[1], self.hidLayerN_))
        S[:, 0, :] = (0.01 - -0.01) * cp.random.rand(X.shape[0], self.hidLayerN_) + -0.01


        SAct, S_unAct= self.calculateStates(RNN_Network.tanh, self.WIH_, self.WHH_, S, Xe)


        O = self.forward(self.WHO_, self.extend(SAct[:, -1, :]))
        Y = self.activation(RNN_Network.softmax, O)

        O = cp.asnumpy(O)
        Y = cp.asnumpy(Y)
        D = cp.asnumpy(D)

        preds = np.argmax(Y, axis=1)

        acc = network.accuracy_np(preds, D)
        loss = RNN_Network.crossEntropyLoss_np(Y, D)

        loss_total = np.sum(loss)

        print("Current Cross-Entropy Loss is: ", loss_total, ", Accuracy is: ", acc)


    
            




    
    







file = h5py.File('data-Mini Project 2.h5', 'r')

training_data = file['trX'][:,:,:]
training_labels = file['trY'][:,:]

test_data = file['tstX'][:,:,:]
test_labels = file['tstY'][:,:]

training_data_shuffled, training_labels_shuffled = shuffle(training_data, training_labels)

print(training_data.shape)

""" d = cp.array([[1, 0, 0], [0, 0, 1]])
y  = cp.array([[0.6, 0.3, 0.1], [0.3, 0.1, 0.6]])

print(RNN_Network.crossEntropyLoss(y, d))

y = cp.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2]])
d  = cp.array([[1, 0, 0], [0, 0, 1]])

print(RNN_Network.calculate_DeDy(y, d)) """

""" X = cp.zeros((2, 3, 4))

Xe = RNN_Network.extend(X)
print(Xe) """


train_subsample = training_data_shuffled[:30,:,:]
train_sublabel = training_labels_shuffled[:30,:]
test_data = cp.asarray(test_data)
test_labels = cp.asarray(test_labels)

######################################################
learning_constant = 0.05
hidden_layer_size = 50
batch_size = 5
time_truncation_length = 10

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)

network.print_timers()

print("Accuracy of 0005_50_10: ", acc)

cp.save("0005_50_10_WHH", network.WHH_)
cp.save("0005_50_10_WHO", network.WHO_)
cp.save("0005_50_10_WIH", network.WIH_)
######################################################
 
""" learning_constant = 0.05
hidden_layer_size = 50
batch_size = 32
time_truncation_length = 30

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)


network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Accuracy of 0005_50_30: ", acc)

cp.save("0005_50_30_WHH", network.WHH_)
cp.save("0005_50_30_WHO", network.WHO_)
cp.save("0005_50_30_WIH", network.WIH_)
###################################################### """

""" learning_constant = 0.05
hidden_layer_size = 100
batch_size = 32
time_truncation_length = 10

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Accuracy of 005_100_10: ", acc)

cp.save("005_100_10_WHH", network.WHH_)
cp.save("005_100_10_WHO", network.WHO_)
cp.save("005_100_10_WIH", network.WIH_)
###################################################### """

""" learning_constant = 0.05
hidden_layer_size = 100
batch_size = 32
time_truncation_length = 30

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Accuracy of 005_100_30: ", acc)

cp.save("005_100_30_WHH", network.WHH_)
cp.save("005_100_30_WHO", network.WHO_)
cp.save("005_100_30_WIH", network.WIH_)
###################################################### """

""" learning_constant = 0.1
hidden_layer_size = 50
batch_size = 32
time_truncation_length = 10

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Accuracy of 0001_50_10: ", acc)

cp.save("0001_50_10_WHH", network.WHH_)
cp.save("0001_50_10_WHO", network.WHO_)
cp.save("0001_50_10_WIH", network.WIH_)
###################################################### """

""" learning_constant = 0.1
hidden_layer_size = 50
batch_size = 32
time_truncation_length = 30

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Accuracy of 0001_50_30: ", acc)

cp.save("0001_50_30_WHH", network.WHH_)
cp.save("0001_50_30_WHO", network.WHO_)
cp.save("0001_50_30_WIH", network.WIH_)
###################################################### """

""" learning_constant = 0.1
hidden_layer_size = 100
batch_size = 32
time_truncation_length = 10

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Accuracy of 001_100_10: ", acc)

cp.save("001_100_10_WHH", network.WHH_)
cp.save("001_100_10_WHO", network.WHO_)
cp.save("001_100_10_WIH", network.WIH_)
###################################################### """

learning_constant = 0.1
hidden_layer_size = 100
batch_size = 32
time_truncation_length = 30

epochs = 50

network = RNN_Network(learning_constant, hidden_layer_size, 3)
network.initWeights()
print("Learning Constant: ", learning_constant, "Hidden Layer Size: ",hidden_layer_size, "Batch Size: ",batch_size, "Time Truncation Length: ",time_truncation_length)

network.run_BPTT(train_subsample, train_sublabel, batch_size, time_truncation_length, epochs)

preds = network.predict(test_data)
acc = network.accuracy_np(cp.asnumpy(preds), cp.asnumpy(test_labels))
print("Test Set Acc: ", acc)
network.print_timers()

print("Test Set Accuracy of 001_100_30: ", acc)

cp.save("001_100_30_WHH", network.WHH_)
cp.save("001_100_30_WHO", network.WHO_)
cp.save("001_100_30_WIH", network.WIH_)
######################################################
