import numpy as np





class Optimizer:

    def __init__(self, learning_rate, momentum_const, checkpoint='') -> None:
        
        if checkpoint is None:
            self.dW1 = 0
            self.dW2 = 0
            self.dW3 = 0 

            self.learning_rate = learning_rate
            self.momentum_const = momentum_const

        else:
            optimizer_states = checkpoint['optimizer_states']
            optimizer_params = checkpoint['optimizer_parameters']

            self.dW1 = optimizer_states['dW1']
            self.dW2 = optimizer_states['dW2']
            self.dW3 = optimizer_states['dW3']

            self.learning_rate = optimizer_params['learning_rate']
            self.momentum_const = optimizer_params['momentum_constant']

        

    def step(self, model, batch):

        self.calculate_forwards(model, batch)
        self.calculate_weight_updates(model, batch)

        model.W3 = model.W3 + self.dW3
        model.W2 = model.W2 + self.dW2
        model.W1 = model.W1 + self.dW1
    

    def states(self):

        optimizer_states = {
            'dW1': self.dW1,
            'dW2': self.dW2,
            'dW3': self.dW3
        }

        return optimizer_states
    

    def parameters(self):

        optimizer_parameters = {
            'learning_rate': self.learning_rate,
            'momentum_constant': self.momentum_const
        }

        return optimizer_parameters



    def calculate_forwards(self, model, batch):

        X = batch['data']

        x_extended = model.extend(X)

        v1 = model.forward(model.W1, x_extended)
        a1 = model.activate_hidden(v1)
        a1_extended = model.extend(a1)

        v2 = model.forward(model.W2, a1_extended, do_dropout=True)
        a2 = model.activate_hidden(v2)
        a2_extended = model.extend(a2)

        v3 = model.forward(model.W3, a2_extended)
        a3 = model.activate_out(v3)

        self.v1 = v1
        self.a1 = a1
        
        self.v2 = v2
        self.a2 = a2

        self.v3 = v3
        self.a3 = a3


    def calculate_weight_updates(self, model, batch):

        DeDW3 = self.calculate_DeDW3(model, batch)
        DeDW2 = self.calculate_DeDW2(model, batch)
        DeDW1 = self.calculate_DeDW1(model, batch)

        self.dW3 = -1 * self.learning_rate * DeDW3 + self.momentum_const * self.dW3
        self.dW2 = -1 * self.learning_rate * DeDW2 + self.momentum_const * self.dW2
        self.dW1 = -1 * self.learning_rate * DeDW1 + self.momentum_const * self.dW1


    def calculate_DeDW3(self, model, batch):

        data = batch['data']

        labels = batch['label']
        labels_onehot = model.oneHotEncode(labels, 6)


        epsilon=1e-6

        Y = self.a3
        dY = -1 * np.reciprocal(Y + epsilon)     
        DeDy3 = dY * labels_onehot

        
        Dy3Dv3 = self.calculate_Dy3Dv3(Y)


        DeDv3 = np.empty((DeDy3.shape[0], Dy3Dv3.shape[2]))
        for i in range(DeDy3.shape[0]):
            DeDv3[i, :] = DeDy3[i, :] @ Dy3Dv3[i, :, :]


        y2 = self.a2
        Dv3DW3 = model.extend(y2)

        DeDW3 = np.einsum('ij,ik->jk', DeDv3, Dv3DW3) / data.shape[0]
        #DeDW3 = np.sum(outer_prods, axis=(1, 3))

        return DeDW3


    def calculate_DeDW2(self, model, batch):

        data = batch['data']

        labels = batch['label']
        labels_onehot = model.oneHotEncode(labels, 6)


        epsilon=1e-6

        Y = self.a3
        dY = -1 * np.reciprocal(Y + epsilon)     
        DeDy3 = dY * labels_onehot

        
        Dy3Dv3 = self.calculate_Dy3Dv3(Y)


        DeDv3 = np.empty((DeDy3.shape[0], Dy3Dv3.shape[2]))
        for i in range(DeDy3.shape[0]):
            DeDv3[i, :] = DeDy3[i, :] @ Dy3Dv3[i, :, :]


        Dv3Dy2 = model.W3
        DeDy2 = DeDv3 @ Dv3Dy2

        v2 = self.v2
        Dy2Dv2 = model.Drelu(model.extend(v2))
        
        DeDv2 = DeDy2 * Dy2Dv2

        y1_extended = model.extend(self.a1)
        Dv2DW2 = y1_extended


        DeDW2 = np.einsum('ij,ik->jk', DeDv2, Dv2DW2)[:-1, :] / data.shape[0]

        return DeDW2
    


    def calculate_DeDW1(self, model, batch):

        data = batch['data']

        labels = batch['label']
        labels_onehot = model.oneHotEncode(labels, 6)

        epsilon=1e-6

        Y = self.a3
        dY = -1 * np.reciprocal(Y + epsilon)     
        DeDy3 = dY * labels_onehot


        Dy3Dv3 = self.calculate_Dy3Dv3(Y)


        DeDv3 = np.empty((DeDy3.shape[0], Dy3Dv3.shape[2]))
        for i in range(DeDy3.shape[0]):
            DeDv3[i, :] = DeDy3[i, :] @ Dy3Dv3[i, :, :]


        Dv3Dy2 = model.W3
        DeDy2 = DeDv3 @ Dv3Dy2

        v2 = self.v2
        Dy2Dv2 = model.Drelu(model.extend(v2))


        DeDv2 = np.empty((v2.shape[0], Dy2Dv2.shape[1]))
        for i in range(v2.shape[0]):
            DeDv2[i, :] = DeDy2[i, :] @ np.diag(Dy2Dv2[i, :])


        Dv2Dy1 = np.concatenate((model.W2, np.zeros((1, model.W2.shape[1]))), axis=0)

        v1 = self.v1

        Dy1Dv1 = model.Drelu(model.extend(v1))

        DeDv1 = (DeDv2 @ Dv2Dy1) * Dy1Dv1

        X = batch['data']
        x_extended = model.extend(X)
        Dv1DW1 = x_extended


        DeDW1 = np.einsum('ij,ik->jk', DeDv1, Dv1DW1)[:-1, :] / data.shape[0]

        return DeDW1 
    

    def calculate_Dy3Dv3(self, Y):
        Dy3Dv3 = np.empty((Y.shape[0], 6, 6))
        for k in range(Y.shape[0]):
            for i in range(6):
                for j in range(6):
                    if i == j:
                        Dy3Dv3[k, i, i] = Y[k, i] * (1 - Y[k, i])
                    else:
                        Dy3Dv3[k, i, j] = -1 * Y[k, i] * Y[k, j]
        return Dy3Dv3





        




        


        



