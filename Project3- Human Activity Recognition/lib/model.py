from numba import njit
import numpy as np
from scipy.special import softmax as softmaxx



class Model:
    def __init__(self, N1, N2, dropout_rate, checkpoint=None, mode='train') -> None:
 

        if checkpoint is not None:
            model_states = checkpoint['model_states']
            model_params = checkpoint['model_parameters']

            self.W1 = model_states['W1']
            self.W2 = model_states['W2']
            self.W3 = model_states['W3']

            self.N1 = model_params['N1']
            self.N2 = model_params['N2']

            self.dropout_rate = model_params['dropout_rate']
        else:
            self.N1 = N1
            self.N2 = N2

            self.dropout_rate = dropout_rate

            self.init_weights()

        self.mode = mode





    def relu(self, x):
        return x * (x > 0)
    
    def Drelu(self, x):
        return (x > 0)
    
    def softmax(self, x):
        return softmaxx(x, axis=1)
    
    def Dsoftmax(self, Y):
        result = np.empty((Y.shape[0], Y.shape[0]))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[0]):
                if i == j:
                    result[i, i] = Y[i] * (1 - Y[i])
                else:
                    result[i, j] = -1 * Y[i] * Y[j]               

        return result

    def forward(self, W, X, do_dropout=False):
        result = np.transpose(W @ np.transpose(X))

        if do_dropout == True and self.mode == 'train':
            dropout_mask = np.random.binomial(1, self.dropout_rate, size=result.shape)
            result = result * dropout_mask        

        return result
    
    def activate(self, func, X):
        result = func(X)
        return result
    
    def activate_hidden(self, X):
        result = self.activate(self.relu, X)
        return result

    def activate_out(self, X):
        result = self.activate(self.softmax, X)
        return result

    def init_weights(self):
        self.W1 = (0.1 - -0.1) * np.random.rand(self.N1, 561 + 1) + -0.1
        self.W1[:, -1] = 0
        self.W2 = (0.1 - -0.1) * np.random.rand(self.N2, self.N1 + 1) + -0.1
        self.W2[:, -1]  = 0
        self.W3 = (0.1 - -0.1) * np.random.rand(6, self.N2 + 1) + -0.1
        self.W3[:, -1]  = 0

    def extend(self, X):
        mOnes = np.expand_dims(np.ones((X.shape[:-1])) * -1, axis=len(X.shape)-1)
        Xe = np.concatenate((X, mOnes), axis=len(X.shape)-1)
        return Xe
    
    def oneHotEncode(self, x, cols):
        zeroMat = np.zeros((x.size, cols))
        zeroMat[np.arange(x.size), x.flatten()] = 1

        return zeroMat
    

    def states(self):

        model_states = {
            'W1': self.W1,
            'W2': self.W2,
            'W3': self.W3 
        }

        return model_states
    


    def parameters(self):

        model_parameters = {
            'N1': self.N1,
            'N2': self.N2,
            'dropout_rate': self.dropout_rate
        }

        return model_parameters
    

    def set_mode(self, mode):

        if mode == 'test' and self.mode == 'train':
            self.W2 = self.W2 * self.dropout_rate
        elif mode == 'train' and self.mode == 'test':
            self.W2 = self.W2 * (1 / self.dropout_rate)

        self.mode = mode



    def batch_loss_acc(self, batch):

        X = batch['data']
        D = batch['label']
        iterator = batch['iterator']
        iterator_max = batch['iteration_max']
        X_extended = self.extend(X)

        v1 = self.forward(self.W1, X_extended)
        a1 = self.activate_hidden(v1)
        a1_extended = self.extend(a1)

        v2 = self.forward(self.W2, a1_extended, do_dropout=True)
        a2 = self.activate_hidden(v2)
        a2_extended = self.extend(a2)

        v3 = self.forward(self.W3, a2_extended)
        y = self.activate_out(v3)


        preds = np.argmax(y, axis=1)
        preds_onehot = self.oneHotEncode(preds, 6)

        desired_onehot = self.oneHotEncode(D, 6)



        epsilon = 1e-6
        batch_loss = -1 * np.sum(self.oneHotEncode(D, 6) * np.log(y+epsilon))
        batch_acc = np.mean(np.sum(preds_onehot * desired_onehot, axis=1))



        return batch_loss, batch_acc
    

    def rateof_listof_mismatches(self, batch):

        X = batch['data']
        D = batch['label']
        iterator = batch['iterator']
        iterator_max = batch['iteration_max']
        X_extended = self.extend(X)

        v1 = self.forward(self.W1, X_extended)
        a1 = self.activate_hidden(v1)
        a1_extended = self.extend(a1)

        v2 = self.forward(self.W2, a1_extended, do_dropout=True)
        a2 = self.activate_hidden(v2)
        a2_extended = self.extend(a2)

        v3 = self.forward(self.W3, a2_extended)
        y = self.activate_out(v3)


        preds = np.argmax(y, axis=1)
        preds_onehot = self.oneHotEncode(preds, 6)

        desired_onehot = self.oneHotEncode(D, 6)

        match_rates = np.around(np.sum(preds_onehot * desired_onehot, axis=0) / np.sum(desired_onehot, axis=0), decimals=4)

        mismatched_pattern_list = []
        matches = np.sum(preds_onehot * desired_onehot, axis=1)
        for i in range(matches.shape[0]):
            if matches[i] == 0:
                mismatched_pattern_list.append(i)
        
        return match_rates, mismatched_pattern_list 
        


        

    

            













    















    




       








 






    

    



        