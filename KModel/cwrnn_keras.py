# https://github.com/braingineer/ikelos/blob/master/ikelos/layers/cwrnn.py
# modified for keras 2.0
# self.U is now self.recurrent_kernel
# self.W is now self.kernel
# self.b is now self.bias
from keras.layers import SimpleRNN
import keras
import numpy as np
import keras.backend as k_backend


class ClockworkRNN(SimpleRNN):
    """
        Clockwork Recurrent Unit - Koutnik et al. 2014

        Clockwork RNN splits simple RNN neurons into groups of equal sizes.
        Each group is activated every specified period. As a result, fast
        groups capture short-term input features while slow groups capture
        long-term input features.

        References:
            A Clockwork RNN
                http://arxiv.org/abs/1402.3511

            others:
            https://github.com/keras-team/keras/issues/2669
            https://github.com/DeanIsMe/Keras_Objects
            https://github.com/dzhu622911/nn-keras-cwrnn
            https://github.com/keras-team/keras/tree/master/keras/layers
            https://github.com/braingineer/ikelos
            https://github.com/dzhu622911/nn-tf-cwrnn
            https://github.com/anirudh9119/clockwork_rnn
            https://cm-gitlab.stanford.edu/tsob/musicNet/tree/02b1bc6ba5d5f299dc80db46be97a07d9fdd222d/theanets-cwRNN

        Modified by liangzulin

    """
    k_v = keras.__version__
    target_v = '2.1.0'

    def __init__(self, units, period_spec=None, debug=False, **kwargs):
        # self.units = units
        if period_spec is None:
            period_spec = [1]
            print("period_spec is None")
        if debug:
            print('$$$$$$$$$$ period_spec == ', period_spec)
        assert units % len(period_spec) == 0, ("Clockwork RNN requires the units to be " +
                                               "a multiple of the number of periods; " +
                                               "units %% len(period_spec) failed.")
        self.period_spec = np.asarray(sorted(period_spec, reverse=True))
        self.mask = None  # <-------------- initialise mask
        self.period = None  # <------------ initialise period
        self.periods = None  # <----------- initialise periods
        self.recurrent_kernel = None  # <-- initialise recurrent_kernel
        self.debug = debug
        super(ClockworkRNN, self).__init__(units=units, **kwargs)

    def build(self, input_shape):
        # construct the clockwork structures
        # basically: every n units the period changes;
        # `period` is for flaggin this; `mask` is for enforcing it
        n = self.units // len(self.period_spec)
        mask = np.zeros((self.units, self.units), k_backend.floatx())
        period = np.zeros((self.units,), np.int16)
        for i, t in enumerate(self.period_spec):
            mask[i * n:(i + 1) * n, i * n:] = 1
            period[i * n:(i + 1) * n] = t
        self.mask = k_backend.variable(mask, name='clockword_mask')
        # self.period = k_backend.variable(period, dtype='int16', name='clockwork_period')

        super(ClockworkRNN, self).build(input_shape)

        # self.U is now self.recurrent_kernel
        # self.W is now self.kernel
        # self.b is now self.bias

        if ClockworkRNN.k_v > ClockworkRNN.target_v:
            self.cell.recurrent_kernel = self.cell.recurrent_kernel * self.mask
        else:
            # old implementation did this at run time
            self.recurrent_kernel = self.recurrent_kernel * self.mask

        # self.periods = self.period
        # self.cell.period = self.period
        # self.cell.periods = self.period

        if self.debug:
            print('After cwrnn build')
            print("self.period_spec--------------------------------")
            print(self.period_spec)
            print("self.mask--------------------------------")
            print(self.mask, type(self.mask))
            print("self.period--------------------------------")
            print(self.period, type(self.period))
            print("self.periods--------------------------------")
            print(self.periods)

            if ClockworkRNN.k_v > ClockworkRNN.target_v:
                print("self.cell.recurrent_kernel--------------------------------")
                print(self.cell.recurrent_kernel)
            else:
                print("self.recurrent_kernel--------------------------------")
                print(self.recurrent_kernel)

        # simple rnn initializes the wrong size self.states
        # we want to also keep the time step in the state.
        if self.stateful:
            if self.debug:
                print("stateful")
            self.reset_states()
        else:
            if self.debug:
                print("not stateful")

            if ClockworkRNN.k_v > ClockworkRNN.target_v:
                self.cell.states = [None, None]
            else:
                self.states = [None, None]

    def get_initial_states(self, x):
        initial_states = super(ClockworkRNN, self).get_initial_states(x)
        if self.debug:
            print("WARNING initial_states == ", initial_states)
            print(type(super(ClockworkRNN, self)), super(ClockworkRNN, self))

        if self.go_backwards:
            input_length = self.input_spec[0].shape[1]
            initial_states[-1] = float(input_length)
        else:
            initial_states[-1] = k_backend.variable(0.)
        return initial_states

    def reset_states(self, **kwargs):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape

        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')

        if self.go_backwards:
            initial_time = self.input_spec[0].shape[1]
        else:
            initial_time = 0.

        if hasattr(self, 'states'):
            if ClockworkRNN.k_v > ClockworkRNN.target_v:
                k_backend.set_value(self.cell.states[0], np.zeros((input_shape[0], self.units)))
                k_backend.set_value(self.cell.states[1], initial_time)
            else:
                k_backend.set_value(self.states[0], np.zeros((input_shape[0], self.units)))
                k_backend.set_value(self.states[1], initial_time)

        else:
            if ClockworkRNN.k_v > ClockworkRNN.target_v:
                self.cell.states = [k_backend.zeros((input_shape[0], self.units)),
                                    k_backend.variable(initial_time)]
            else:
                self.states = [k_backend.zeros((input_shape[0], self.units)),
                               k_backend.variable(initial_time)]

    def get_constants(self, x, training=None):
        if self.debug:
            print('WARING: get_constants method has been call ---> line 106')
        # consts = super(ClockworkRNN, self).get_constants(x, training=None)
        consts = super(ClockworkRNN, self).get_constants(x, training=training)
        consts.append(self.period)
        return consts

    def step(self, x, states):
        prev_output = states[0]
        time_step = states[1]
        b__u = states[2]
        b__w = states[3]
        period = states[4]

        if self.debug:
            print(type(b__w))
            print(prev_output, time_step, b__u, b__w, period)

        h = x

        if ClockworkRNN.k_v > ClockworkRNN.target_v:
            output = self.activation(h + k_backend.dot(prev_output * b__u,
                                                       self.cell.recurrent_kernel))
        else:
            output = self.activation(h + k_backend.dot(prev_output * b__u,
                                                       self.recurrent_kernel))

        if k_backend.backend() == 'tensorflow':
            import tensorflow as tf
            output = tf.where(k_backend.equal(tf.cast(time_step, tf.float32) % tf.cast(period, tf.float32),
                                              0.),
                              output,
                              prev_output)
        else:
            output = k_backend.switch(k_backend.equal(time_step % period,
                                                      0.),
                                      output,
                                      prev_output)

        return output, [output, time_step + 1]

    def get_config(self):
        config = {"periods": self.periods}
        base_config = super(ClockworkRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
