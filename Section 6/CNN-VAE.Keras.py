# Importing the libraries
import numpy as np  # Importing NumPy for numerical computations
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape  # Importing Keras layers for building the model
from keras.models import Model  # Importing the Model class to define the neural network
from keras import backend as K  # Importing Keras backend for low-level operations
from keras.callbacks import EarlyStopping  # Importing EarlyStopping callback to prevent overfitting

# Setting the input dimensions (size of frames)
INPUT_DIM = (64,64,3)  # Defining the input shape as 64x64 RGB images (height, width, channels)

# Setting the number of convolutional filters, kernel sizes, strides, and activations per layer
CONV_FILTERS = [32,64,64,128]  # Number of filters for each convolutional layer in the encoder
CONV_KERNEL_SIZES = [4,4,4,4]  # Kernel sizes for each convolutional layer in the encoder
CONV_STRIDES = [2,2,2,2]  # Strides for each convolutional layer in the encoder
CONV_ACTIVATIONS = ['relu','relu','relu','relu']  # Activation functions for each convolutional layer in the encoder

# Setting the dense layer size
DENSE_SIZE = 1024  # Number of units in the dense layer after flattening

# Setting the layer parameters for the decoder part of the VAE
CONV_T_FILTERS = [64,64,32,3]  # Number of filters for each transposed convolutional layer in the decoder
CONV_T_KERNEL_SIZES = [5,5,6,6]  # Kernel sizes for each transposed convolutional layer in the decoder
CONV_T_STRIDES = [2,2,2,2]  # Strides for each transposed convolutional layer in the decoder
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']  # Activation functions for each transposed convolutional layer in the decoder

# Setting the dimensions of the latent vectors
Z_DIM = 32  # Dimensionality of the latent space (output of the encoder)

# Setting the number of epochs and batch size
EPOCHS = 1  # Number of times the model will iterate over the entire dataset
BATCH_SIZE = 32  # Number of samples processed before updating the model’s weights

# Making a function that creates centralized latent vectors for the VAE
def sampling(args):
    z_mean, z_log_var = args  # Unpacking the mean and log variance of the latent distribution
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0., stddev=1.)  # Sampling random noise from a standard normal distribution
    return z_mean + K.exp(z_log_var / 2) * epsilon  # Applying the reparameterization trick to sample from the latent space

# Building the CNN-VAE model within a class
class ConvVAE:

    # Initializing all the parameters and variables of the ConvVAE class
    def __init__(self):
        self.models = self._build()  # Building the VAE model, encoder, and decoder
        self.model = self.models[0]  # The full VAE model
        self.encoder = self.models[1]  # The encoder part of the VAE
        self.decoder = self.models[2]  # The decoder part of the VAE
        self.input_dim = INPUT_DIM  # Storing the input dimensions
        self.z_dim = Z_DIM  # Storing the latent space dimensionality

    # Building the model
    def _build(self):
        # Creating the model and the encoder inputs
        vae_x = Input(shape=INPUT_DIM)  # Defining the input tensor for the model

        # Creating the first convolutional layer of the Encoder
        vae_c1 = Conv2D(filters=CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZES[0], strides=CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)  # First convolutional layer

        # Creating the second convolutional layer of the Encoder
        vae_c2 = Conv2D(filters=CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZES[1], strides=CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)  # Second convolutional layer

        # Creating the third convolutional layer of the Encoder
        vae_c3 = Conv2D(filters=CONV_FILTERS[2], kernel_size=CONV_KERNEL_SIZES[2], strides=CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)  # Third convolutional layer

        # Creating the fourth convolutional layer of the Encoder
        vae_c4 = Conv2D(filters=CONV_FILTERS[3], kernel_size=CONV_KERNEL_SIZES[3], strides=CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)  # Fourth convolutional layer

        # Flattening the last convolutional layer so we can input it in the dense layers
        vae_z_in = Flatten()(vae_c4)  # Flattening the output of the last convolutional layer

        # Using two separate files to calculate z_mean and z_log
        vae_z_mean = Dense(Z_DIM)(vae_z_in)  # Dense layer to compute the mean of the latent distribution
        vae_z_log_var = Dense(Z_DIM)(vae_z_in)  # Dense layer to compute the log variance of the latent distribution

        # Using the Lambda Keras class around the sampling function we created above
        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])  # Applying the sampling function to generate latent vectors

        # Getting the inputs of the decoder part
        vae_z_input = Input(shape=(Z_DIM,))  # Defining the input tensor for the decoder

        # Instantiating these layers separately so as to reuse them later
        vae_dense = Dense(1024)  # Dense layer for the decoder
        vae_dense_model = vae_dense(vae_z)  # Applying the dense layer to the latent vector

        # Reshaping the dense layer to 4 dimensions, so we can put it through the transposed convolution
        vae_z_out = Reshape((1,1,DENSE_SIZE))  # Reshape layer to prepare for transposed convolution
        vae_z_out_model = vae_z_out(vae_dense_model)  # Applying the reshape layer

        # Defining the first transposed convolutional layer
        vae_d1 = Conv2DTranspose(filters=CONV_T_FILTERS[0], kernel_size=CONV_T_KERNEL_SIZES[0], strides=CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])  # First transposed convolutional layer
        vae_d1_model = vae_d1(vae_z_out_model)  # Applying the first transposed convolutional layer

        # Defining the second transposed convolutional layer
        vae_d2 = Conv2DTranspose(filters=CONV_T_FILTERS[1], kernel_size=CONV_T_KERNEL_SIZES[1], strides=CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])  # Second transposed convolutional layer
        vae_d2_model = vae_d2(vae_d1_model)  # Applying the second transposed convolutional layer

        # Defining the third convolutional layer
        vae_d3 = Conv2DTranspose(filters=CONV_T_FILTERS[2], kernel_size=CONV_T_KERNEL_SIZES[2], strides=CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])  # Third transposed convolutional layer
        vae_d3_model = vae_d3(vae_d2_model)  # Applying the third transposed convolutional layer

        # Defining the fourth convolutional layer
        vae_d4 = Conv2DTranspose(filters=CONV_T_FILTERS[3], kernel_size=CONV_T_KERNEL_SIZES[3], strides=CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])  # Fourth transposed convolutional layer
        vae_d4_model = vae_d4(vae_d3_model)  # Applying the fourth transposed convolutional layer

        # Getting the latent vector output of the decoder
        vae_dense_decoder = vae_dense(vae_z_input)  # Applying the dense layer to the decoder input
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)  # Applying the reshape layer to the decoder input
        vae_d1_decoder = vae_d1(vae_z_out_decoder)  # Applying the first transposed convolutional layer to the decoder input
        vae_d2_decoder = vae_d2(vae_d1_decoder)  # Applying the second transposed convolutional layer to the decoder input
        vae_d3_decoder = vae_d3(vae_d2_decoder)  # Applying the third transposed convolutional layer to the decoder input
        vae_d4_decoder = vae_d4(vae_d3_decoder)  # Applying the fourth transposed convolutional layer to the decoder input

        # Defining the end-to-end VAE Model, composed of both the encoder and the decoder
        vae = Model(vae_x, vae_d4_model)  # Full VAE model (encoder + decoder)
        vae_encoder = Model(vae_x, vae_z)  # Encoder model
        vae_decoder = Model(vae_z_input, vae_d4_decoder)  # Decoder model

        # Implementing the training operations
        # Defining the MSE loss
        def vae_r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)  # Flattening the true values
            y_pred_flat = K.flatten(y_pred)  # Flattening the predicted values
            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)  # Computing the mean squared error (MSE) loss

        # Defining the KL divergence loss
        def vae_kl_loss(y_true, y_pred):
            return -0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)  # Computing the KL divergence loss

        # Defining the total VAE loss, summing the MSE and KL losses
        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)  # Total VAE loss (reconstruction + KL divergence)

        # Compiling the whole model with the RMSProp optimizer, the vae loss, and custom metrics
        vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])  # Compiling the model with the specified loss and metrics

        return (vae, vae_encoder, vae_decoder)  # Returning the VAE model, encoder, and decoder

    # Loading the model
    def set_weights(self, filepath):
        self.model.load_weights(filepath)  # Loading pre-trained weights into the model

    # Creating early stopping callbacks to prevent overfitting
    def train(self, data, validation_split=0.2):
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')  # Defining the EarlyStopping callback
        callbacks_list = [earlystop]  # Creating a list of callbacks
        self.model.fit(data, data,  # Training the model
                       shuffle=True,  # Shuffling the data
                       epochs=EPOCHS,  # Number of epochs
                       batch_size=BATCH_SIZE,  # Batch size
                       validation_split=validation_split,  # Validation split
                       callbacks=callbacks_list)  # Adding callbacks
        self.model.save_weights('vae/weights.h5')  # Saving the model weights

    # Saving the model
    def save_weights(self, filepath):
        self.model.save_weights(filepath)  # Saving the model weights to a file

    # Generating data for the MDN-RNN
    def generate_rnn_data(self, obs_data, action_data):
        rnn_input = []  # List to store RNN inputs
        rnn_output = []  # List to store RNN outputs
        for i, j in zip(obs_data, action_data):  # Iterating over observation and action data
            rnn_z_input = self.encoder.predict(np.array(i))  # Encoding observations into latent vectors
            conc = [np.concatenate([x,y]) for x, y in zip(rnn_z_input, j.reshape(-1, 1))]  # Concatenating latent vectors with actions
            rnn_input.append(conc[:-1])  # Appending RNN inputs
            rnn_output.append(np.array(rnn_z_input[1:]))  # Appending RNN outputs
        rnn_input = np.array(rnn_input)  # Converting RNN inputs to a NumPy array
        rnn_output = np.array(rnn_output)  # Converting RNN outputs to a NumPy array
        print("Rnn inputs size: {}".format(rnn_input.shape), " Rnn outputs size: {}".format(rnn_output.shape))  # Printing the shapes of RNN inputs and outputs
        return (rnn_input, rnn_output)  # Returning RNN inputs and outputs