    
if __name__ == '__main__':


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    #from scipy.special import softmax as softmaxx
    from cupyx.scipy.special import softmax as softmaxx
    import h5py
    from sklearn.utils import shuffle
    import cupy as cp
    import multiprocessing 
    import concurrent



    expit = cp.ElementwiseKernel(
        'float64 x',
        'float64 y',
        'y = 1 / (1 + exp(-x))',
        'expit')

    class RNN_Network:

        def __init__(self, learnConst, hidLayerN, timeTruncationLength, batchSize, validationSize) -> None:
            self.learning_constant_ = learnConst
            self.hidLayerN_ = hidLayerN
            self.time_truncation_length_ = timeTruncationLength
            self.batch_size_ = batchSize
            self.validation_size_ = validationSize
            

        def tanh(x):
            return cp.tanh(x)
        
        def Dtanh(x):
            return 1 - cp.square(RNN_Network.tanh(x)) 
        
        def softmax(x):
            return softmaxx(x)
        
        def Dsoftmax(x):
            return RNN_Network.softmax(x) * (1 - RNN_Network.softmax(x))
        
        def extend(self, X):
            mOnes = cp.expand_dims(cp.ones((X.shape[:-1])) * -1, axis=len(X.shape)-1)
            Xe = cp.concatenate((X, mOnes), axis=len(X.shape)-1)
            return Xe
        
        def calculate_DeDy_sd(self, Y, D):
            epsilon=1e-6
            dY = -1 * cp.reciprocal(Y+epsilon)     
            result = dY * D

            return result
        
        def calculate_DyDv_sd(self, Y):
            result = cp.empty((Y.shape[0], Y.shape[0]))
            for i in range(Y.shape[0]):
                for j in range(Y.shape[0]):
                    if i == j:
                        result[i, i] = Y[i] * (1 - Y[i])
                    else:
                        result[i, j] = -1 * Y[i] * Y[j]               

            return result
        
        
        def calculate_DvDh_sd(self, WHO):
            return WHO
        
        def calculate_DhiDvi_sd(self, hi, dForward):
            result = dForward
            return result

        def calculate_DviDhip_sd(self, WHH):
            return WHH
        
        def calculate_DhiDhip_sd(self, hi, dForwardi, WHH):
            DhiDvi = self.calculate_DhiDvi_sd(hi, dForwardi)
            DviDhip = self.calculate_DviDhip_sd(WHH)

            DhiDhip = DhiDvi[:, cp.newaxis] * DviDhip
            return DhiDhip

        def calculate_DhiDhip_tensor_sd(self, hi_matrix, dForward_matrix, WHH):

            DhiDhip_tensor = cp.empty((hi_matrix.shape[0], hi_matrix.shape[1], hi_matrix.shape[1]))
            DhiDhip_tensor[0,:,:] = cp.eye(hi_matrix.shape[1])

            for i in range(1, DhiDhip_tensor.shape[0]):
                DhiDhip = self.calculate_DhiDhip_sd(hi_matrix[i-1, :], dForward_matrix[i-1, :], WHH)

                norm_constant = 1
                norm_term = norm_constant / cp.linalg.norm(DhiDhip)


                DhiDhip_tensor[i,:,:] = norm_term * DhiDhip

            return DhiDhip_tensor
        

        def calculate_DhDhi_tensor_sd(self, hi_matrix, dForward_matrix, WHH):

            DhiDhip_tensor = self.calculate_DhiDhip_tensor_sd(hi_matrix, dForward_matrix, WHH)

            DhDhi_tensor = cp.empty((hi_matrix.shape[0], hi_matrix.shape[1], hi_matrix.shape[1]))
            for i in range(0, hi_matrix.shape[0]):

                product = cp.eye(hi_matrix.shape[1])
                for j in range(i):
                    if (cp.linalg.norm(product) > 3):
                        product = product * (3 / (cp.linalg.norm(product) + 1e-6))
                    product = product @ DhiDhip_tensor[j, :, :]
                DhDhi_tensor[i, :, :] = product

            return DhiDhip_tensor
        

        def calculate_DhDvi_sd(self, hi_matrix, dForward_matrix, WHH, i):

            DhDhi_tensor = self.calculate_DhDhi_tensor_sd(hi_matrix, dForward_matrix, WHH)
            DhDhi = DhDhi_tensor[i, :, :]

            DhiDvi = cp.diag(self.calculate_DhiDvi_sd(hi_matrix[i, :], dForward_matrix[i, :]))

            DhDvi = DhDhi @ DhiDvi

            zero_row = cp.zeros((1, DhDvi.shape[1]))
            DhDvi_extended = cp.concatenate((DhDvi, zero_row), axis=0)

            return DhDvi_extended


        def calculate_DeDvi_sd(self, Y, D, WHO, hi_matrix, dForward_matrix, WHH, i):
            
            DeDy = self.calculate_DeDy_sd(Y, D)
            DyDv = self.calculate_DyDv_sd(Y)
            DvDh = self.calculate_DvDh_sd(WHO)
            DhDvi = self.calculate_DhDvi_sd(hi_matrix, dForward_matrix, WHH, i)

            DeDvi = DeDy @ DyDv @ DvDh @ DhDvi

            return DeDvi
        


        def calculate_DvDWHO_sd_flatted(self, h_last_extended):
            result = cp.vstack([h_last_extended] * 6)
            return result
        
        def calculate_DviDWHH_sd_flatted(self, hi_previous):
            result = cp.vstack([hi_previous] * hi_previous.size)
            return result
        
        def calculate_DviDWIH_sd_flatted(self, xi):
            result = cp.vstack([xi] * self.hidLayerN_)
            return result




        def calculate_DeDWHH_sd_timei(self, Y, D, WHO, hi_matrix, dForward_matrix, WHH, i):
            DeDvi = self.calculate_DeDvi_sd(Y, D, WHO, hi_matrix, dForward_matrix, WHH, i)
            DviDWHH_flatted = self.calculate_DviDWHH_sd_flatted(hi_matrix[i+1])

            DeDWHH = DeDvi[:, cp.newaxis] * DviDWHH_flatted
            return DeDWHH
        
        def calculate_DeDWIH_sd_timei(self, Y, D, WHO, hi_matrix, dForward_matrix, WHH, X_matrix, i):
            DeDvi = self.calculate_DeDvi_sd(Y, D, WHO, hi_matrix, dForward_matrix, WHH, i)
            DviDWIH_flatted = self.calculate_DviDWIH_sd_flatted(X_matrix[i+1])

            DeDWIH = DeDvi[:, cp.newaxis] * DviDWIH_flatted
            return DeDWIH
        
        def calculate_DeDWHH_sd_total(self, Y, D, WHO, hi_matrix, dForward_matrix, WHH):

            curTotal = cp.zeros(WHH.shape)
            for i in range(hi_matrix.shape[0]-1):
                curTotal = curTotal + self.calculate_DeDWHH_sd_timei(Y, D, WHO, hi_matrix, dForward_matrix, WHH, i)

            return curTotal
        
        def calculate_DeDWIH_sd_total(self, Y, D, WHO, hi_matrix, dForward_matrix, WHH, X_matrix):

            curTotal = cp.zeros((WHH.shape[0], 4))
            for i in range(hi_matrix.shape[0]-1):
                curTotal = curTotal + self.calculate_DeDWIH_sd_timei(Y, D, WHO, hi_matrix, dForward_matrix, WHH, X_matrix, i)

            return curTotal

        def calculate_DeDWHO_sd(self, Y, D, h_last_extended):
            DeDy = self.calculate_DeDy_sd(Y, D)
            DyDv = self.calculate_DyDv_sd(Y)

            DeDv = DeDy @ DyDv

            DvDWHO_flatted = self.calculate_DvDWHO_sd_flatted(h_last_extended)

            DeDWHO = DeDv[:, cp.newaxis] * DvDWHO_flatted

            return DeDWHO
        
        def forwardInputandState_sd_st(self, x, h, WIH, WHH):
            v_x = WIH @ x 
            v_h = WHH @ h
            v = v_x + v_h

            return v
        
        def activate(self, v, func):
            return func(v)
        

        def calculate_both_states_sd(self, X_matrix, h_initial, WIH, WHH):

            h_matrix_activated = cp.empty((X_matrix.shape[0], WHH.shape[0]))
            h_matrix_activated[-1, :] = h_initial

            h_matrix_unactivated = h_matrix_activated

            for i in reversed(range(X_matrix.shape[0])):
                v = self.forwardInputandState_sd_st(X_matrix[X_matrix.shape[0] -i-1, :], h_matrix_activated[i, :], WIH, WHH) ##Buraya yara bandı attım.
                h_next = self.activate(v, RNN_Network.tanh)

                h_matrix_unactivated[i-1, :]  = v
                h_matrix_activated[i-1, :] = h_next
            
            return h_matrix_activated, h_matrix_unactivated
        

        def predict_sd(self, h_last_extended, WHO):
            v = WHO @ h_last_extended
            y = self.activate(v, RNN_Network.softmax)

            return y
        

        def calculate_states_AndPredict_sd(self, X_matrix, h_inital, WIH, WHH, WHO):
            h_matrix_activated, h_matrix_unactivated = self.calculate_both_states_sd(X_matrix, h_inital, WIH, WHH)

            h_last = h_matrix_activated[0, :]
            h_last_extended = self.extend(h_last)

            y = self.predict_sd(h_last_extended, WHO)

            return y, h_matrix_activated, h_matrix_unactivated
        

        def calculate_weight_changes_sd(self, Y, D, WHO, WHH, hi_activated_matrix, hi_unactivated_matrix, X_matrix, learning_constant):

            h_last = hi_activated_matrix[0, :]
            h_last_extended = self.extend(h_last)

            hi_Dactivated_matrix = self.activate(hi_unactivated_matrix, RNN_Network.Dtanh)

            DeDWHO = self.calculate_DeDWHO_sd(Y, D, h_last_extended)
            DeDWHH = self.calculate_DeDWHH_sd_total(Y, D, WHO, hi_activated_matrix, hi_Dactivated_matrix, WHH)
            DeDWIH = self.calculate_DeDWIH_sd_total(Y, D, WHO, hi_activated_matrix, hi_Dactivated_matrix, WHH, X_matrix)

            dWHO = -1 * learning_constant * DeDWHO
            dWHH = -1 * learning_constant * (DeDWHH / X_matrix.shape[0])
            dWIH = -1 * learning_constant * (DeDWIH / X_matrix.shape[0])

            return dWHO, dWHH, dWIH



        def forward_network_and_calculate_weight_changes(self, X_matrix, h_inital, WIH, WHH, WHO, d, learning_constant):

            y, h_matrix_activated, h_matrix_unactivated = self.calculate_states_AndPredict_sd(X_matrix, h_inital, WIH, WHH, WHO)
            dWHO, dWHH, dWIH = self.calculate_weight_changes_sd(y, d, WHO, WHH, h_matrix_activated, h_matrix_unactivated, X_matrix, learning_constant)

            h_last = h_matrix_activated[0, :]

            return dWHO, dWHH, dWIH, h_last


        def run_BPTT_on_truncated(self, X_matrix_tensor, d_matrix, h_inital_matrix, WHO, WHH, WIH):

            new_h_initial_matrix = cp.empty(h_inital_matrix.shape)


            dWHO_sum = 0
            dWHH_sum = 0
            dWIH_sum = 0

            for i in range(X_matrix_tensor.shape[0]):
                current_X_matrix = X_matrix_tensor[i, :, :]
                current_h_initial = h_inital_matrix[i, :]
                d_current = d_matrix[i, :]
                
                dWHO, dWHH, dWIH, h_last = self.forward_network_and_calculate_weight_changes(current_X_matrix, current_h_initial, WIH, WHH, WHO, d_current, self.learning_constant_)

                dWHO_sum = dWHO_sum + dWHO
                dWHH_sum = dWHH_sum + dWHH
                dWIH_sum = dWIH_sum + dWIH

                new_h_initial_matrix[i, :] = h_last

            self.WHO_ = self.WHO_ + dWHO_sum
            self.WHH_ = self.WHH_ + dWHH_sum
            self.WIH_ = self.WIH_ + dWIH_sum

            return new_h_initial_matrix


        def run_BPTT_on_minibatch_wholetime(self, X_matrix_tensor_wholetime, d_matrix, time_truncation_length): #####DANGER!!!

            N_time = X_matrix_tensor_wholetime.shape[1] // time_truncation_length

            X_matrix_tensor_timeparts = cp.array_split(X_matrix_tensor_wholetime, N_time, axis=1) 

            h_initial_matrix = (0.01 - -0.01) * cp.random.rand(X_matrix_tensor_wholetime.shape[0], self.hidLayerN_) + -0.01

            for x_matrix_tensor_timeparts_current in X_matrix_tensor_timeparts:
                new_h_initial_matrix = self.run_BPTT_on_truncated(x_matrix_tensor_timeparts_current, d_matrix, h_initial_matrix, self.WHO_, self.WHH_, self.WIH_)
                h_initial_matrix = new_h_initial_matrix
            

            """ executor = concurrent.futures.ProcessPoolExecutor(N_time)
            futures = [executor.submit(self.run_BPTT_on_truncated(x_matrix_tensor_timeparts_current, d_matrix, h_initial_matrix, self.WHO_, self.WHH_, self.WIH_)) for x_matrix_tensor_timeparts_current in X_matrix_tensor_timeparts]
            concurrent.futures.wait(futures) """



        def run_BPTT_oneEpoch(self, X_matrix_tensor_whole, d_matrix):
            
            N_samples = X_matrix_tensor_whole.shape[0] // self.batch_size_

            X_matrix_tensor_sampleparts = cp.array_split(X_matrix_tensor_whole, N_samples, axis=0)
            d_matrix_sampleparts = cp.array_split(d_matrix, N_samples, axis=0)

            
            executor = concurrent.futures.ProcessPoolExecutor(N_samples)
            futures = [executor.submit(self.run_BPTT_on_minibatch_wholetime(x_matrix_tensor_sampleparts_current, d_matrix_sampleparts_current, self.time_truncation_length_), (x_matrix_tensor_sampleparts_current, d_matrix_sampleparts_current)) for x_matrix_tensor_sampleparts_current, d_matrix_sampleparts_current in zip(X_matrix_tensor_sampleparts, d_matrix_sampleparts)]
            concurrent.futures.wait(futures)
        

            """ for x_matrix_tensor_sampleparts_current, d_matrix_sampleparts_current in zip(X_matrix_tensor_sampleparts, d_matrix_sampleparts):
                self.run_BPTT_on_minibatch_wholetime(x_matrix_tensor_sampleparts_current, d_matrix_sampleparts_current, self.time_truncation_length_) """



        def run_BPTT(self, X_matrix_tensor_whole, d_matrix, epochs):

            #X_matrix_tensor_whole = cp.flip(X_matrix_tensor_whole, axis=1)

            X_matrix_tensor_whole_extended = self.extend(X_matrix_tensor_whole)

            X_matrix_tensor_whole_train = X_matrix_tensor_whole_extended[:-self.validation_size_, :, :]
            X_matrix_tensor_whole_validation = X_matrix_tensor_whole_extended[-self.validation_size_:, : :]

            d_matrix_train = d_matrix[:-self.validation_size_, :]
            d_matrix_validation = d_matrix[-self.validation_size_: :]


            print("Model Hyperparameters: Learning Constant={lc}, Hidden N={N}, Time Length={tl}, Batch Size={bs}, Validation Size={vl}".format(lc=self.learning_constant_, N=self.hidLayerN_, tl=self.time_truncation_length_, bs=self.batch_size_, vl=self.validation_size_))
            

            for i in range(epochs):

                print("Starting epoch {epoch_num}...".format(epoch_num=i+1))


                self.run_BPTT_oneEpoch(X_matrix_tensor_whole_train, d_matrix_train)


                y_matrix_train = self.forward_and_predict(X_matrix_tensor_whole_train)
                y_matrix_validation = self.forward_and_predict(X_matrix_tensor_whole_validation)

                train_loss = RNN_Network.crossEntropyLoss(y_matrix_train, d_matrix_train)
                validation_loss = RNN_Network.crossEntropyLoss(y_matrix_validation, d_matrix_validation)

                train_acc = self.accuracy(y_matrix_train, d_matrix_train)
                validation_acc = self.accuracy(y_matrix_validation, d_matrix_validation)

                mismatched_patterns_rate_validation = self.mismatched_patterns_rate(y_matrix_validation, d_matrix_validation)

                print("Epoch {epoch_num} done:".format(epoch_num=i+1))
                print("Current Loss on Training Set: {train_loss}, Current Accuracy on Training Set: {train_acc}".format(train_loss=train_loss, train_acc=train_acc))
                
                
                if self.validation_size_ > 0:
                    print("Current Loss on Validation Set: {validation_loss}, Current Accuracy on Validation Set: {validation_acc}".format(validation_loss=validation_loss, validation_acc=validation_acc))
                    print("Current pattern match-rates on Validation Set: {mm_rate}".format(mm_rate=mismatched_patterns_rate_validation))


        def forward_and_predict(self, X_matrix_tensor_whole):

            y_matrix = cp.empty((X_matrix_tensor_whole.shape[0], 6))
            
            for i in range(X_matrix_tensor_whole.shape[0]):
                h_initial = (0.01 - -0.01) * cp.random.rand(1, self.hidLayerN_) + -0.01

                y_current, _, _ = self.calculate_states_AndPredict_sd(X_matrix_tensor_whole[i, :, :], h_initial, self.WIH_, self.WHH_, self.WHO_)

                y_matrix[i, :] = y_current

            return y_matrix

        
        def accuracy(self, Y_matrix, D_matrix):

            Y_matrix_cpu = cp.asnumpy(Y_matrix)
            D_matrix_cpu = cp.asnumpy(D_matrix)

            acc = np.mean(np.argmax(Y_matrix_cpu, axis=1) == np.argmax(D_matrix_cpu, axis=1))
            return acc
        
        def crossEntropyLoss(Y_matrix, D_matrix):
            epsilon = 1e-6
            return -1 * np.sum(D_matrix * np.log(Y_matrix+epsilon))

        def initWeights(self):
            self.WIH_ = (0.01 - -0.01) * cp.random.rand(self.hidLayerN_, 3+1) + -0.01
            self.WIH_[-1,:] = 0
            self.WHH_ = (0.01 - -0.01) * cp.random.rand(self.hidLayerN_, self.hidLayerN_) + -0.01
            self.WHH_[-1,:] = 0
            self.WHO_ = (0.01 - -0.01) * cp.random.rand(6, self.hidLayerN_ +1) + -0.01
            self.WHO_[-1,:] = 0

        def mismatched_patterns_rate(self, y_matrix, d_matrix):
            epsilon = 1e-6
            y_matrix_onehot = self.y_max_onehot_top_k(y_matrix, 1)
            result = cp.round(cp.sum(y_matrix_onehot * d_matrix, axis=0) / (cp.sum(d_matrix, axis=0) + epsilon), decimals=2)
            return result
        
        def y_max_onehot_top_k(self, y_matrix, k):

            y_matrix_current = y_matrix
            y_max_onehot_final = cp.zeros(y_matrix.shape)

            for i in range(k):
                max_y_current = cp.asarray(np.argmax(cp.asnumpy(y_matrix_current), axis=1))
                max_y_onehot_current = RNN_Network.oneHotEncode(max_y_current, 0)
                y_max_onehot_final = y_max_onehot_final + max_y_onehot_current
                y_max_current_onehot = y_matrix_current * max_y_onehot_current
                y_matrix_next = y_matrix_current - y_max_current_onehot
                y_matrix_current = y_matrix_next
            
            return y_max_onehot_final
        

        def top_k_accuracy(self, y_max_matrix, d_matrix, k):

            y_max_matrix_onehot_topk = self.y_max_onehot_top_k(y_max_matrix, k)

            result = cp.round(y_max_matrix_onehot_topk * d_matrix, decimals=2)
            return result

        def test_network(self, x_matrix_test, d_matrix_test):

            x_matrix_test_extended = self.extend(x_matrix_test)

            y_matrix_test = self.forward_and_predict(x_matrix_test_extended)
            test_loss = RNN_Network.crossEntropyLoss(y_matrix_test, d_matrix_test)
            mm_pattern_rates = self.mismatched_patterns_rate(y_matrix_test, d_matrix_test)
            print("Loss on Test Set: {loss}, Pattern Match Rates: {mm}".format(loss=test_loss, mm=mm_pattern_rates))
            
            for k in range(1, 5):
                top_k_acc = cp.sum(cp.mean(self.top_k_accuracy(y_matrix_test, d_matrix_test, k), axis=0))
                print("Top {k} Accuracy on Test Set: {acc}".format(k=k, acc=top_k_acc))

        def oneHotEncode(x, l):
            zeroMat = cp.zeros((x.size, 6)) + l
            zeroMat[cp.arange(x.size), x] = 1

            return zeroMat
        

        



        



    





    file = h5py.File('data-Mini Project 2.h5', 'r')

    training_data = file['trX'][:,:,:]
    training_labels = file['trY'][:,:]

    test_data = file['tstX'][:,:,:]
    test_labels = file['tstY'][:,:]

    training_data_shuffled, training_labels_shuffled = shuffle(training_data, training_labels)

    print(training_data.shape)


    ###################################################################################

    training_data_subset = cp.asarray(training_data_shuffled[:50, :20, :])
    training_labels_subset = cp.asarray(training_labels_shuffled[:50, :])

    test_data_GPU = cp.array(test_data)
    test_labels_GPU = cp.array(test_labels)


    learning_constant = 0.1
    hidden_layer = 50
    time_truncation_length = 10
    batch_size = 1
    validation_size = 10
    epochs= 10

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("005_50_10_WHO", network.WHO_)
    np.save("005_50_10_WHH", network.WHH_)
    np.save("005_50_10_WIH", network.WIH_)
    
    ###################################################################################


    """ learning_constant = 0.05
    hidden_layer = 50
    time_truncation_length = 30
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("005_50_30_WHO", network.WHO_)
    np.save("005_50_30_WHH", network.WHH_)
    np.save("005_50_30_WIH", network.WIH_)


    ###################################################################################


    learning_constant = 0.1
    hidden_layer = 50
    time_truncation_length = 10
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("01_50_10_WHO", network.WHO_)
    np.save("01_50_10_WHH", network.WHH_)
    np.save("01_50_10_WIH", network.WIH_)


    ###################################################################################


    learning_constant = 0.1
    hidden_layer = 50
    time_truncation_length = 30
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("01_50_30_WHO", network.WHO_)
    np.save("01_50_30_WHH", network.WHH_)
    np.save("01_50_30_WIH", network.WIH_)


    ###################################################################################


    learning_constant = 0.05
    hidden_layer = 100
    time_truncation_length = 10
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("005_100_10_WHO", network.WHO_)
    np.save("005_100_10_WHH", network.WHH_)
    np.save("005_100_10_WIH", network.WIH_)


    ###################################################################################


    learning_constant = 0.05
    hidden_layer = 100
    time_truncation_length = 30
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("005_100_30_WHO", network.WHO_)
    np.save("005_100_30_WHH", network.WHH_)
    np.save("005_100_30_WIH", network.WIH_)


    ###################################################################################



    learning_constant = 0.1
    hidden_layer = 100
    time_truncation_length = 10
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("01_100_10_WHO", network.WHO_)
    np.save("01_100_10_WHH", network.WHH_)
    np.save("01_100_10_WIH", network.WIH_)


    ###################################################################################



    learning_constant = 0.1
    hidden_layer = 100
    time_truncation_length = 10
    batch_size = 5
    validation_size = 300
    epochs= 50

    network = RNN_Network(learning_constant, hidden_layer, time_truncation_length, batch_size, validation_size)
    network.initWeights()
    network.run_BPTT(training_data_subset, training_labels_subset, epochs)
    network.test_network(test_data_GPU, test_labels_GPU)
    np.save("01_100_10_WHO", network.WHO_)
    np.save("01_100_10_WHH", network.WHH_)
    np.save("01_100_10_WIH", network.WIH_) """





        



        

        
        
        







    


        



    










        


    

    












    
   

    
    


    


    
