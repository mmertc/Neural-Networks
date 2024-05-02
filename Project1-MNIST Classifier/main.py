import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import idx2numpy as idx
from scipy.special import expit
from scipy.special import softmax as softmaxx
import random
from timeit import default_timer as timer


class MLClassifier:
    def __init__(self, learnConst, hidLayerN, funcHidden, funcHiddenD, funcOut, FuncOutD) -> None:
        self.learnConst_ = learnConst
        self.hidLayerN_ = hidLayerN
        self.funcHidden_ = funcHidden
        self.funcHiddenD_ = funcHiddenD
        self.funcOut_ = funcOut
        self.FuncOutD_ = FuncOutD
        

    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return x * (x > 0)
    
    def sigmoid(self, x):
        return expit(x)
    
    def softmax(self, x):
        return softmaxx(x, axis=1)



    def forward(self, W, X):
        Xe = MLClassifier.extendInput(X)
        return np.transpose(W @ np.transpose(Xe))
    
    def activation(self, func, Z):
        return func(self, Z)
    
    def applyLayer(self, func, W, X):
        Z = self.forward(W, X)
        Y = self.activation(func, Z)
        return Y
    
    def predict(self, funcHidden, funcOut, W1, W2, X):
        Y = self.applyLayer(funcHidden, W1, X)
        O = self.applyLayer(funcOut, W2, Y)

        preds = np.argmax(O, axis=1)

        return preds
    
    def accuraricy(self, preds, truth):
        acc = np.mean(preds == truth)
        return acc

    def initWeights(self):
        self.W1_ = (0.01 - -0.01) * np.random.rand(self.hidLayerN_, 28*28+1) + -0.01
        self.W1_[-1,:] = 0
        self.W2_ = (0.01 - -0.01) * np.random.rand(10, self.hidLayerN_ +1) + -0.01
        self.W2_[-1,:] = 0

    def errorL2Half(self, out, desired):
        err = out - desired
        return 1/2 * np.linalg.norm(err)

    def Dtanh(self, x):
        return 1 - np.square(self.tanh(x)) 
    
    def Drelu(self, x):
        return (x > 0)

    def Dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def Dsoftmax(self, x):
        return self.softmax(x) * (1 - self.softmax(x))
    
    
    def applyNetworkAndSave(self, funcHidden, funcHiddenD, funcOut, funcOutD, W1, W2, X, D, lambd=0):
        self.F1_ = self.forward(W1, X)
        self.A1_ = self.activation(funcHidden, self.F1_)

        self.F2_ = self.forward(W2, self.A1_)
        self.A2_ = self.activation(funcOut, self.F2_)

        E = D - self.A2_

        self.lg2_ = np.multiply(E, funcOutD(self, self.F2_))
        self.lg1_ = np.multiply(funcHiddenD(self, self.F1_), np.transpose(np.transpose(W2) @ np.transpose(self.lg2_))[:, :-1])


        Dcost2 = np.einsum('ik,kj->ij', np.transpose(self.lg2_), MLClassifier.extendInput(self.A1_)) - X.shape[0] * lambd * W2
        Dcost1 = np.einsum('ik,kj->ij', np.transpose(self.lg1_), MLClassifier.extendInput(X)) - X.shape[0] * lambd * W1

        self.dW2_ = self.learnConst_ * Dcost2
        self.dW1_ = self.learnConst_ * Dcost1


    def runBackPropagation_(self, batch, D, lambd=0):
        self.applyNetworkAndSave(self.funcHidden_, self.funcHiddenD_, self.funcOut_, self.FuncOutD_, self.W1_, self.W2_, batch, D, lambd) 
        
        self.W1_ = self.W1_ + self.dW1_ / batch.shape[0]
        self.W2_ = self.W2_ + self.dW2_ / batch.shape[0]



    def runBackPropagation(self, epochs, batchSize, X, D, lambd=0):  

        n = X.shape[0] / batchSize

        batches = np.array_split(X, n, axis=0)
        desireds = np.array_split(D, n, axis=0)
        
        for k in range(epochs):
            for i, j in zip(batches, desireds):  
                self.runBackPropagation_(i, j, lambd)


    def extendInput(X):
        rows = X.shape[0]
        ones = np.ones((rows, 1))
        return np.concatenate((X, -1 * ones), axis=1)
    

    def oneHotEncode(x, l):
        zeroMat = np.zeros((x.size, 10)) + l
        zeroMat[np.arange(x.size), x] = 1

        return zeroMat

    


train_data = idx.convert_from_file('train-images.idx3-ubyte')
train_label = idx.convert_from_file('train-labels.idx1-ubyte')
test_data = idx.convert_from_file('t10k-images.idx3-ubyte')
test_label = idx.convert_from_file('t10k-labels.idx1-ubyte')



train_data_f = train_data.reshape(60000, -1) / 255
train_label_f = train_label
test_data_f = test_data.reshape(10000, -1) / 255
test_label_f = test_label


desiredsB = MLClassifier.oneHotEncode(train_label_f, -1)
desiredsU = MLClassifier.oneHotEncode(train_label_f, 0)

inputs_train = train_data_f
inputs_test = test_data_f








network = MLClassifier(0.01, 300, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_300_tanh_tanh_W1", network.W1_)
np.save("001_300_tanh_tanh_W2", network.W2_)

print("N=300, lc= 0.01, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 300, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_300_tanh_tanh_W1", network.W1_)
np.save("005_300_tanh_tanh_W2", network.W2_)


print("N=300, lc= 0.05, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 300, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_300_tanh_tanh_W1", network.W1_)
np.save("009_300_tanh_tanh_W2", network.W2_)


print("N=300, lc= 0.09, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.01, 500, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_500_tanh_tanh_W1", network.W1_)
np.save("001_500_tanh_tanh_W2", network.W2_)


print("N=500, lc= 0.01, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 500, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_500_tanh_tanh_W1", network.W1_)
np.save("005_500_tanh_tanh_W2", network.W2_)


print("N=500, lc= 0.05, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 500, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_500_tanh_tanh_W1", network.W1_)
np.save("009_500_tanh_tanh_W2", network.W2_)


print("N=500, lc= 0.09, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.01, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_1000_tanh_tanh_W1", network.W1_)
np.save("001_1000_tanh_tanh_W2", network.W2_)


print("N=1000, lc= 0.01, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_1000_tanh_tanh_W1", network.W1_)
np.save("005_1000_tanh_tanh_W2", network.W2_)


print("N=1000, lc= 0.05, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsB)
end= timer()
preds = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_1000_tanh_tanh_W1", network.W1_)
np.save("009_1000_tanh_tanh_W2", network.W2_)


print("N=1000, lc= 0.09, tanh, tanh; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)





network = MLClassifier(0.01, 300, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid) ###############
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_300_relu_sigmoid_W1", network.W1_)
np.save("001_300_relu_sigmoid_W2", network.W2_)


print("N=300, lc= 0.01, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)





network = MLClassifier(0.01, 500, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_500_relu_sigmoid_W1", network.W1_)
np.save("001_500_relu_sigmoid_W2", network.W2_)


print("N=500, lc= 0.01, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.01, 1000, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_1000_relu_sigmoid_W1", network.W1_)
np.save("001_1000_relu_sigmoid_W2", network.W2_)


print("N=1000, lc= 0.01, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 300, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_300_relu_sigmoid_W1", network.W1_)
np.save("005_300_relu_sigmoid_W2", network.W2_)


print("N=300, lc= 0.05, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 500, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_500_relu_sigmoid_W1", network.W1_)
np.save("005_500_relu_sigmoid_W2", network.W2_)


print("N=500, lc= 0.05, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 1000, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_1000_relu_sigmoid_W1", network.W1_)
np.save("005_1000_relu_sigmoid_W2", network.W2_)


print("N=1000, lc= 0.05, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 300, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_300_relu_sigmoid_W1", network.W1_)
np.save("009_300_relu_sigmoid_W2", network.W2_)


print("N=300, lc= 0.09, relu, sigmoid; Acc: ", acc, " ,L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 500, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_500_relu_sigmoid_W1", network.W1_)
np.save("009_500_relu_sigmoid_W2", network.W2_)


print("N=500, lc= 0.09, relu, sigmoid; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 1000, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.sigmoid, MLClassifier.Dsigmoid)
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.sigmoid, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_1000_relu_sigmoid_W1", network.W1_)
np.save("009_1000_relu_sigmoid_W2", network.W2_)


print("N=1000, lc= 0.09, relu, sigmoid; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)







network = MLClassifier(0.01, 300, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) ###############
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_300_relu_softmax_W1", network.W1_)
np.save("001_300_relu_softmax_W2", network.W2_)


print("N=300, lc= 0.01, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)






network = MLClassifier(0.01, 500, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_500_relu_softmax_W1", network.W1_)
np.save("001_500_relu_softmax_W2", network.W2_)


print("N=500, lc= 0.01, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)



network = MLClassifier(0.01, 1000, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("001_1000_relu_softmax_W1", network.W1_)
np.save("001_1000_relu_softmax_W2", network.W2_)


print("N=1000, lc= 0.01, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)


network = MLClassifier(0.05, 300, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_300_relu_softmax_W1", network.W1_)
np.save("005_300_relu_softmax_W2", network.W2_)


print("N=300, lc= 0.05, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 500, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_500_relu_softmax_W1", network.W1_)
np.save("005_500_relu_softmax_W2", network.W2_)


print("N=500, lc= 0.05, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.05, 1000, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("005_1000_relu_softmax_W1", network.W1_)
np.save("005_1000_relu_softmax_W2", network.W2_)


print("N=1000, lc= 0.05, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)




network = MLClassifier(0.09, 300, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_300_relu_softmax_W1", network.W1_)
np.save("009_300_relu_softmax_W2", network.W2_)


print("N=300, lc= 0.09, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)



network = MLClassifier(0.09, 500, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_500_relu_softmax_W1", network.W1_)
np.save("009_500_relu_softmax_W2", network.W2_)


print("N=500, lc= 0.09, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)



network = MLClassifier(0.09, 1000, MLClassifier.relu, MLClassifier.Drelu, MLClassifier.softmax, MLClassifier.Dsoftmax) 
network.initWeights()

start = timer()
network.runBackPropagation(100, train_data_f.shape[0], inputs_train, desiredsU)
end= timer()
preds = network.predict(MLClassifier.relu, MLClassifier.softmax, network.W1_, network.W2_, inputs_test)
acc = network.accuraricy(preds, test_label_f)
L2HalfErrAvg = network.errorL2Half(preds, test_label_f) / test_label_f.shape[0]
np.save("009_1000_relu_softmax_W1", network.W1_)
np.save("009_1000_relu_softmax_W2", network.W2_)


print("N=1000, lc= 0.09, relu, softmax; Acc: ", acc, ", L2HalfErrAvg: ", L2HalfErrAvg, ", CPU time: ", end-start)





network = MLClassifier(0.09, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, 30, inputs_train, desiredsB)
end= timer()
predstest = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acctest = network.accuraricy(predstest, test_label_f)
L2HalfErrAvgtest = network.errorL2Half(predstest, test_label_f) / test_label_f.shape[0]
predstrain = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_train)
acctrain = network.accuraricy(predstrain, train_label_f)
L2HalfErrAvgtrain = network.errorL2Half(predstrain, train_label_f) / train_label_f.shape[0]
np.save("009_1000_tanh_tanh_batchsize30_W1", network.W1_)
np.save("009_1000_tanh_tanh_batchsize30_W2", network.W2_)


print("N=1000, lc= 0.09, tanh, tanh, batch-size: 100; AccTest: ", acctest, "AccTrain: ", acctrain, " ,L2HalfErrAvgTest: ", L2HalfErrAvgtest, ",L2HalfErrAvgTrain: ", L2HalfErrAvgtrain, ", CPU time: ", end-start)


network = MLClassifier(0.09, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, 50, inputs_train, desiredsB)
end= timer()
predstest = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acctest = network.accuraricy(predstest, test_label_f)
L2HalfErrAvgtest = network.errorL2Half(predstest, test_label_f) / test_label_f.shape[0]
predstrain = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_train)
acctrain = network.accuraricy(predstrain, train_label_f)
L2HalfErrAvgtrain = network.errorL2Half(predstrain, train_label_f) / train_label_f.shape[0]
np.save("009_1000_tanh_tanh_batchsize50_W1", network.W1_)
np.save("009_1000_tanh_tanh_batchsize50_W2", network.W2_)


print("N=1000, lc= 0.09, tanh, tanh, batch-size: 100; AccTest: ", acctest, "AccTrain: ", acctrain, " ,L2HalfErrAvgTest: ", L2HalfErrAvgtest, ",L2HalfErrAvgTrain: ", L2HalfErrAvgtrain, ", CPU time: ", end-start)




network = MLClassifier(0.09, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh)
network.initWeights()

start = timer()
network.runBackPropagation(100, 100, inputs_train, desiredsB)
end= timer()
predstest = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acctest = network.accuraricy(predstest, test_label_f)
L2HalfErrAvgtest = network.errorL2Half(predstest, test_label_f) / test_label_f.shape[0]
predstrain = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_train)
acctrain = network.accuraricy(predstrain, train_label_f)
L2HalfErrAvgtrain = network.errorL2Half(predstrain, train_label_f) / train_label_f.shape[0]
np.save("009_1000_tanh_tanh_batchsize100_W1", network.W1_)
np.save("009_1000_tanh_tanh_batchsize100_W2", network.W2_)


print("N=1000, lc= 0.09, tanh, tanh, batch-size: 100; AccTest: ", acctest, "AccTrain: ", acctrain, " ,L2HalfErrAvgTest: ", L2HalfErrAvgtest, ",L2HalfErrAvgTrain: ", L2HalfErrAvgtrain, ", CPU time: ", end-start)



network = MLClassifier(0.09, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh) ########
network.initWeights()

start = timer()
network.runBackPropagation(100, 10, inputs_train, desiredsB, lambd=0.01)
end= timer()
predstest = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acctest = network.accuraricy(predstest, test_label_f)
L2HalfErrAvgtest = network.errorL2Half(predstest, test_label_f) / test_label_f.shape[0]
predstrain = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_train)
acctrain = network.accuraricy(predstrain, train_label_f)
L2HalfErrAvgtrain = network.errorL2Half(predstrain, train_label_f) / train_label_f.shape[0]
np.save("009_1000_tanh_tanh_batchsize10_lambd001_W1", network.W1_)
np.save("009_1000_tanh_tanh_batchsize10_lambd001_W2", network.W2_)


print("N=1000, lc= 0.09, tanh, tanh, batch-size: 10, lambda=0.01; AccTest: ", acctest, "AccTrain: ", acctrain, " ,L2HalfErrAvgTest: ", L2HalfErrAvgtest, ",L2HalfErrAvgTrain: ", L2HalfErrAvgtrain, ", CPU time: ", end-start)





network = MLClassifier(0.09, 1000, MLClassifier.tanh, MLClassifier.Dtanh, MLClassifier.tanh, MLClassifier.Dtanh) ########
network.initWeights()

start = timer()
network.runBackPropagation(100, 10, inputs_train, desiredsB, lambd=0.001)
end= timer()
predstest = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_test)
acctest = network.accuraricy(predstest, test_label_f)
L2HalfErrAvgtest = network.errorL2Half(predstest, test_label_f) / test_label_f.shape[0]
predstrain = network.predict(MLClassifier.tanh, MLClassifier.tanh, network.W1_, network.W2_, inputs_train)
acctrain = network.accuraricy(predstrain, train_label_f)
L2HalfErrAvgtrain = network.errorL2Half(predstrain, train_label_f) / train_label_f.shape[0]
np.save("009_1000_tanh_tanh_batchsize10_lambd0001_W1", network.W1_)
np.save("009_1000_tanh_tanh_batchsize10_lambd0001_W2", network.W2_)


print("N=1000, lc= 0.09, tanh, tanh, batch-size: 10, lambda=0.001; AccTest: ", acctest, "AccTrain: ", acctrain, " ,L2HalfErrAvgTest: ", L2HalfErrAvgtest, ",L2HalfErrAvgTrain: ", L2HalfErrAvgtrain, ", CPU time: ", end-start)


