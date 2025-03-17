####################
# MDN-RNN
####################
# the code is similar to the structure of the CNN-VAE
####################

# Importing Libraries
import numpy as np
import tensorflow as tf

# Building the MDN-RNN model within a class
##########################


class MDN_RNN(object):

    # Init all params and vars of the MDN-RNN Class
    # the default values are arbitrary
    def __init__(self, hps, reuse=False, gpu_mode=False):

        # this is to initialize the variables
        self.hps = hps                              # hps : hyperparameters for the model
        # reuse : for the variable scope of TF, True if we will reuse the Variable scope
        self.reuse = reuse

        with tf.variable_scope('MDN_RNN', reuse=self.reuse):
            if not gpu_mode:                       # gpu_mode: to toggle use of gpu for training
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu')
                    self.g = tf.Graph()

                    with self.g.as_default():       # Connecting all hyperparameters to the model
                        self._build_model(hps)
            else:
                # you don't need to set the device, because the default is to use the gpu
                tf.logging.info('Model using gpu')
                with self.g.as_default():       # Connecting all hyperparameters to the model
                    self._build_model(hps)

        self._init_session()

    def _build_model(self, hps):
        # Build the model of the MDN-RNN
        # this is where we build the model, and define the loss function
        # and the optimizer for training

        #  Building the RNN
        # num_mixture: number of mixture components in the MDN
        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture  # KMIX: number of mixture components in the MDN
        # input_Width: width of the input sequence (in this case it is the z_size from the CNN-VAE, 32,
        # added to the size of the action, 3, giving a total size of 35)
        input_Width = self.hps.input_seq_width

        # output_Width: width of the output sequence (in this case it is the z_size from the CNN-VAE, 32)
        output_Width = self.hps.output_seq_width

        # This is the max # of steps (in this case 1000)
        length = self.hps.max_seq_length

        # This is if we are in the training mode
        if hps.is_training:
            # Defining the global step of gradient descent, this is the same as the one in the CNN-VAE
            # this is to reduce the loss with the optimized gradient descent
            self.global_step = tf.Variable(
                initial_value=0, name='Global_step', trainable=False)
            # Initial value : this is the first step
            # Trainable : If True, also adds the variable to the graph collection GraphKeys.TRAINABLE_VARIABLES.
            # This collection is used as the default list of variables to use by the Optimizer classes. Defaults to
            # True, unless synchronization is set to ON_READ, in which case it defaults to False.

        # Actually building the RNN
        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
        # cell_fn: this is the function to build the RNN cell, in this case it is a LSTM cell
        # LayerNormBasicLSTMCell: this is a LSTM cell with layer normalization
        # if this = 0 then we won't apply a dropout, otherwise we will
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        # if this = 0 then we won't apply a dropout, otherwise we will
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        # if this = 0 then we won't apply a dropout, otherwise we will
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        # if this = 0 then we won't apply a layer normalization, otherwise we will
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True

        # step 1
        # Create the RNN cell with the specified configurations:
        # - If recurrent dropout is enabled, create the cell with dropout and layer normalization.
        # - Otherwise, create the cell with layer normalization only.
        # - If input dropout is enabled, wrap the cell with DropoutWrapper to apply dropout to the input.
        # - If output dropout is enabled, wrap the cell with DropoutWrapper to apply dropout to the output.
        if use_recurrent_dropout:
            cell = cell_fn(num_units=hps.rnn_size, layer_norm=use_layer_norm,
                           dropout_keep_prop=self.hps.recurrent_dropout_prob)
            # num_units: number of units in the LSTM cell
            # layer_norm: if True, then apply layer normalization to the LSTM cell
            # dropout_keep_prop: the probability of keeping the dropout, if 0 then no dropout is applied

        else:
            cell = cell_fn(num_units=hps.rnn_size, layer_norm=use_layer_norm)
            # we keep all the units activated, so we don't have a dropout

        if use_input_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=cell, input_keep_prob=self.hps.input_dropout_prob)
            # cell: the RNN cell to wrap with dropout
            # input_keep_prob: the probability of keeping the input units active (i.e., not dropped out)

        if use_output_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=cell, output_keep_prob=self.hps.output_dropout_prob)
            # cell: the RNN cell to wrap with dropout
            # output_keep_prob: the probability of keeping the output units active (i.e., not dropped out)

        # step 2
        # create a new obj var that is the LSTM cell with the dropout added or not
        self.cell = cell

        # step 3
        # Wrapping all of the RNN with the inputs and outputs using tf.nn.dynamic_rnn function

        # creating an object var for the max sequence length
        self.sequence_lengths = length

        # We start with tf placeholders for the inputs, targets, and outputs
        # we will use tf.placeholder
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[
                                      self.hps.batch_size, self.hps.max_seq_len, self.hps.input_seq_width])
        # dtype: data type of the input, in this case, float32
        # shape: shape of the input tensor
        # - batch_size: number of sequences in a batch
        # - max_seq_len: maximum length of the input sequences
        # - input_seq_width: width of each input vector in the sequence

        self.output_x = tf.placeholder(dtype=tf.float32, shape=[
                                       self.hps.batch_size, self.hps.max_seq_length, self.hps.output_seq_width])
        # dtype: data type of the target, in this case, float32
        # shape: shape of the target tensor
        # - batch_size: number of sequences in a batch
        # - max_seq_length: maximum length of the output sequences
        # - output_seq_width: width of each output vector in the sequence

        # the output will be done later

        # Now to gather all of the inputs for the final RNN
        actual_input_x = self.input_x
        # actual_input_x: this is the input to the RNN, it is the same as the input_x

        # Creating the initial state with all 0s
        self.initial_state = cell.zero_state(
            batch_size=self.hps.batch_size, dtype=tf.float32)
        # initial_state: this is the initial state of the RNN, it is a zero state
        # batch_size: number of sequences in a batch
        # dtype: data type of the state, in this case, float32

        # step 4
        # making tensors of weights and biases for the output layer
        n_out = output_Width * KMIX * 3  # Number of columns
        # n_out: the number of output units, calculated as the product of output_Width, KMIX, and 3

        with tf.variable_scope('RNN'):
            output_w = tf.get_variable(name='output_w', shape=[
                                       self.hps.rnn_size, n_out])
            # output_w: weights for the output layer
            # name: name of the variable
            # shape: shape of the weights tensor, [rnn_size, n_out]
            output_b = tf.get_variable(name='output_b', shape=[n_out])
            # output_b: biases for the output layer
            # name: name of the variable
            # shape: shape of the biases tensor, [n_out]

        # step 5 : getting deterministic output from the RNN using the tf.nn.dynamic_rnn function
            # we are getting z and h

        # z: the output of the RNN (output)
        # h: the state of the RNN (last_state)
        output, last_state = tf.nn.dynamic_rnn(
            cell=cell, inputs=actual_input_x, initial_state=self.initial_state, dtype=tf.float32, swap_memory=True, scope='RNN')
        # cell: the RNN cell to use
        # inputs: the input tensor to the RNN
        # initial_state: the initial state of the RNN
        # dtype: data type of the inputs and outputs, in this case, float32
        # swap_memory: whether to swap memory from GPU to CPU during training
        # scope: variable scope for the RNN

        # Build the MDN

        # First we will reshape the output of the RNN
        # we gotta flatten the output
        # this is the input of the MDN
        output = tf.reshape(output, [-1, hps.rnn_size])
        # output: the output tensor of the RNN
        # shape: shape of the output tensor, [-1, hps.rnn_size]
        # -1: infers the size of the dimension from the remaining dimensions
        # hps.rnn_size: number of units in the RNN cell

        # Getting the Hidden Layer of the MDN
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        # output: the output tensor after applying the weights and biases
        # tf.nn.xw_plus_b: computes the sum of the matrix multiplication of 'output' and 'output_w' and the bias 'output_b'
        # output_w: weights for the output layer
        # output_b: biases for the output layer

        # reshaping the output again to get the final stochastic output of the MDN-RNN
        output = tf.reshape(output, [-1, KMIX*3])
        # output: the reshaped output tensor
        # shape: shape of the output tensor, [-1, KMIX*3]
        # -1: infers the size of the dimension from the remaining dimensions
        # KMIX*3: the number of mixture components times 3 (for logmix, mean, and logstd)

        # Getting MDN Coefficients
        def get_mdn_coef(output):
            # introducing the variables that represent the splits from the tf.split function
            logmix, mean, logstd = tf.split(output, 3, 1)
            # logmix: the log of the mixture coefficients
            # mean: the mean of the Gaussian distributions
            # logstd: the log of the standard deviation of the Gaussian distributions
            logmix = logmix - \
                tf.reduce_logsumexp(input_tensor=logmix, axis=1, keepdims=True)
            # input_tensor: the input tensor to the reduce_logsumexp function
            # axis: the axis along which to perform the operation
            # keepdims: whether to keep the reduced dimensions in the output tensor
            return logmix, mean, logstd

        out_logmix, out_mean, out_logstd = get_mdn_coef(output)
        # out_logmix: the log of the mixture coefficients
        # out_mean: the mean of the Gaussian distributions
        # out_logstd: the log of the standard deviation of the Gaussian distributions

        self.out_logmix = out_logmix
        # self.out_logmix: the log of the mixture coefficients stored as a class attribute
        self.out_mean = out_mean
        # self.out_mean: the mean of the Gaussian distributions stored as a class attribute
        self.out_logstd = out_logstd
        # self.out_logstd: the log of the standard deviation of the Gaussian distributions stored as a class attribute

        # Implementing Training operations
