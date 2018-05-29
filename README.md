# cwrnn_for_keras_2_1_above

This is a modified version of braingineer's ikelos cwrnn, update it to suit both keras 2.1.4 version and 2.0.4 version

The main change is that adding a keras version detector for selecting suitable function. 
E.g. 

        if ClockworkRNN.k_v > ClockworkRNN.target_v:
            self.cell.recurrent_kernel = self.cell.recurrent_kernel * self.mask
        else:
            self.recurrent_kernel = self.recurrent_kernel * self.mask

And add a simple help how to use it:

    model = Sequential()
    model.add(ClockworkRNN(units=90,
                           period_spec=[1, 2, 4, 8, 16],
                           input_shape=train_x.shape[1:],  # ---(samples, timesteps, dimension)
                           dropout_W=0.4,
                           return_sequences=True,
                           debug=cwrnn_debug))  # debug is for developing mode, you can remove
    model.add(Dropout(.2))
    model.add(TimeDistributed(Dense(units=1, activation='linear')))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=epochs, batch_size=1, verbose=1)

The test code has been run at Windows 7 with Anaconda 2, Keras 2.1.4, Theano.
And it also has been run at Cent OS 7 with Anaconda 2, Keras 2.0.4, Theano.

Predict Result:
![Training and predict curve](https://github.com/liangzulin/cwrnn_for_keras_2_1_above/blob/master/test_cwrnn_1.png?raw=true)

Training and Test curve for epochs from 1 to 100:
![Training and predict curve](https://github.com/liangzulin/cwrnn_for_keras_2_1_above/blob/master/test_cwrnn_2.png?raw=true)
