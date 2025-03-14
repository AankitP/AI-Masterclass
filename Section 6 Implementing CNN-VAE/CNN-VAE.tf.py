###########################
# TensorFlow Version
###########################

# Importing the Libraries
import numpy as np
import tensorflow as tf

# Building the VAE class within a model

##########################
# Building a VAE Model
###########################
class ConvVAE(object):

    # Init all params and vars of the ConvVae Class
    # the default values are arbitrary
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
        # Keep the following in mind  
            # Formula Z = μ + σ * ε
            # z : Stochastic AE latent vector
            # μ : mean vector
            # σ : Standard Deviation
            # ε : Stochastic Node

        # this is to initialize the variables
        self.z_size = z_size                # z_size : size of the latent vector
        self.batch_size = batch_size        # batch_size : Size of the training batches (for batch learning)
        self.learning_rate = learning_rate  # learning_rate : rate of the learning of the CNN-VAE
        self.kl_tolerance = kl_tolerance    # kl_tolerance : this is used to compute the kl-Loss
                                                # the loss we use to train the VAE will be the sum of the mean squared error loss, and the KL-Loss
        self.is_training = is_training      # is_training : mode switcher, to switch in and out of training mode
        self.reuse = reuse                  # reuse : for the variable scope of TF, True if we will reuse the Variable scope
        
        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode :               # gpu_mode: to toggle use of gpu for training
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu')
                    self._build_graph()
            else:
                # you don't need to set the device, because the default is to use the gpu
                tf.logging.info('Model using gpu')
                self._build_graph()
        self._init_session()
    
    # Writing the method that creates the VAE model architecture
    def _build_graph(self):
        # the first thing we will do before making the architecture is to init the graph
            # dynamic graphs allow faster computations of the gradients of composition functions 
                # in the training computation
        self.g = tf.Graph()    

        # The second thing we will do is specify that we want to have the whole architecture
            # of the VAE model inside the graph
        with self.g.as_default():
            # This is where the architecture of the VAE model goes

            # Since we are following the architecture from worldmodels
            # We will be creating the placeholder to be in the format 
            # of the input images : 64 x 64 x 3
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])
                # For the shape there are 4 elements
                    # First : Dimension of the batch in this case we specify none
                    # Rest : Dimension of the images
                        # The reason the last value is 3 is because we are working 
                        # with color images, if it was BW then we would use 1
            
############# Building the Encoder of the VAE #############

            # First convolution layer : Relu conv
            h = tf.layers.conv2d(inputs = self.x , filters = 32, 
                                 kernel_size = 4, strides = 2, 
                                 activation = tf.nn.relu, name = "enc_conv1")
                # For h we call the tf.layers.conv2d function, with the arguments:
                    # inputs : This is the input image
                    # filters : this is the # of feature maps/filters
                        # Since we are following the architecture from worldmodels
                        # we will set this to 32 
                    # kernel_size : this is the size of the feature maps
                    # strides : stride of the feature maps
                    # activation : Activation function to use
                    # name : name of this layer
            
            # 2nd conv layer : relu
                # For each layer we adjust the activiation function, 
                # num of filters, and filter size based on the architecture
            h = tf.layers.conv2d(inputs = h , filters = 64, 
                                 kernel_size = 4, strides = 2, 
                                 activation = tf.nn.relu, name = "enc_conv2")
            
            # 3rd conv layer : relu
            h = tf.layers.conv2d(inputs = h , filters = 128, 
                                 kernel_size = 4, strides = 2, 
                                 activation = tf.nn.relu, name = "enc_conv3")
            
            # 4th conv layer : relu
            h = tf.layers.conv2d(inputs = h , filters = 256, 
                                 kernel_size = 4, strides = 2, 
                                 activation = tf.nn.relu, name = "enc_conv4")
            
            # The next step is to flatten the result of the 
            # convolutions into a 1d vector of 2x2x256 elements
            h = tf.reshape(tensor = h, shape = [-1, 2 * 2 * 256])
                # tensor : this is h (the result of all of the convolutions)
                # shape : This is the shape of the flattened tensor
                    # we set -1 as the first index of shape to specify we want 
                        # it to be a single column vector
                    # 2*2*256 is the num of elements we want in the flattened vector
            
            # Building the V part of the VAE
                # Adding Stochasticity
                # We will sample from a factored Gaussian function
                    # We will do this by making 2 separate fully connected layers 
                        # using the dense function mu and sigma
                        # Dense is the name of the class/function to build 
                            # fully connected layers
                    # then using the random normal distribution by tensorflow
                    # we will sample some nums from a normal dist
                    # then we will add them to get z

            # this is to create the first fully connected layer (mu)
            self.mu = tf.layers.dense(inputs = h, units = self.z_size, name = "enc_fc_mu")
                # dense function will take 3 variables
                    # inputs : the input vector (in this case the flattened vector)
                    # units : number of neurons (based on architecture)

            # this is to create the second fully connected layer (sigma)
            self.logvar = tf.layers.dense(inputs = h, units = self.z_size, name = "enc_fc_logvar")
            self.sigma = tf.exp(self.logvar/2.0)

            # getting the Normal dist of mean 0 and variance 1
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])

            # getting the batch of latent vector z's
            self.z = self.mu + self.sigma * self.epsilon

############# Building the Decoder for the VAE #############

            # First ting we need to do is make another fully connected layer
            h = tf.layers.dense(inputs = self.z, units = 1024, name = "dec_fc")
                # units is 1024 because that is the size of the output
            
            # Now we reshape to a 1d vector of 1x1x1024 size
            h = tf.reshape(input = h, shape = [-1 , 1 , 1 , 1024])
                # the reason for this style of dimension in shape, is to 
                # make sure that we match perfectly with the architecture
            
            # now we invert the convolutions
            h = tf.layers.conv2d_transpose(inputs = h , filters = 128, 
                                 kernel_size = 5, strides = 2, 
                                 activation = tf.nn.relu, name = "dec_deconv1" )
                # this function is basically the same as the conv2d function, 
                    # the only difference is that it inverses convolutions
            # repeat this for the other deconvolutions, adjusting values 
                # according to the architecture
            h = tf.layers.conv2d_transpose(inputs = h , filters = 64, 
                                 kernel_size = 5, strides = 2, 
                                 activation = tf.nn.relu, name = "dec_deconv2" )
            h = tf.layers.conv2d_transpose(inputs = h , filters = 32, 
                                 kernel_size = 6, strides = 2, 
                                 activation = tf.nn.relu, name = "dec_deconv3" )
            # The following is the final reconstruction of the image
                # from the output of the CNN-VAE
            self.y = tf.layers.conv2d_transpose(inputs = h , filters = 3, 
                                 kernel_size = 6, strides = 2, 
                                 activation = tf.nn.sigmoid, name = "dec_deconv4" )

############# Implementing the training operations #############
    # to find and reduce the loss between the prediction (self.y) and target (self.x)
        # the loss we will use is the sum of the Mean squarred error loss a,d the KL (Kullback-Leibler) loss

            # We need to make sure we are in training (we don't want it to run all of the time)
            if (self.is_training):
                # Defining the global step of gradient descent
                    # this is to reduce the loss with the optimized gradient descent
                self.global_step = tf.Variable(initial_value=0, name='Global_step', trainable=False)
                    # Initial value : this is the first step
                    # Trainable : If True, also adds the variable to the graph collection GraphKeys.TRAINABLE_VARIABLES.
                        # This collection is used as the default list of variables to use by the Optimizer classes. Defaults to 
                        # True, unless synchronization is set to ON_READ, in which case it defaults to False.
                
                # Now we will find the losses
                # MSE loss:
                self.r_loss = tf.reduce_sum(input_tensor = tf.square(self.x - self.y), reduction_indecies = [1,2,3]) # Calculate the sum of squared differences between the target and predictions
                    # input_tensor : tensor to reduce
                    # reduction_indecies : the dimensons to reduce (since this is a color image we use [1,2,3], 
                        # if it was BW then we would use [1,2])
                        # we do not start with 0 because the first index refers to the batch in the way we have it setup
                self.r_loss = tf.reduce_mean(self.r_loss)
                    # this computes the mean of the self.r_loss
                
                # KL Loss:
                self.kl_loss = -.5 * tf.reduce_sum(input_tensor=(1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indecies = 1)
                    # -.5 * (1 + {values in the logvar dense layer} - {values in the mu dense layer}^2 - {exponential of the values in the logvar dense layer}
                    # tf.exp : computes the exponential of each element in a tensor. The exponential function is e^x where e is Eulers number
                    # we use reduction_indecies = 1 because the dense layer only has 2 dimensions, the first corresponding to the batch
                        # and the second one corresponding to the 1D vertical tensor
                # Now we will take the max of the computed KL loss and the (kl_tolerance * z_size)
                self.kl_loss =  tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size) 
                    # the reason to use the max of the kl_loss and the (kl_tolerance * z_size) is because if the loss is too small, we won't have to apply the gradient
                        # this is just a trick to make it easier for us to determine if we need to apply the gradient
                # we will now compute the mean of the kl_loss if the max of the 2 is (kl_tolerance * z_size) then the mean will just be (kl_tolerance * z_size)
                    # if it is self.kl_loss, then it will take the mean of the sums of the values of the dense layers
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                # Now we will sum the MSE loss and the kl loss
                self.loss = self.r_loss + self.kl_loss

                # Set learning rate
                self.lr = tf.Variable(initial_value=self.learning_rate, trainable=False)

                # Now we will set variable for the optimizer (we are using the adam optimizer) for stochastic gradient descent
                self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
                    # this class is already implemented in tf, and only takes the learning rate as an argument
                
                # Now we have everything to compute the gradient



                # Computing the Gradient on the loss (the result of the compute gradient method of the Adam Optimizer)
                grads = self.optimizer.compute_gradients(loss=self.loss)
                    # Loss : loss 
                
                # Now we will apply the gradient on the loss to reduce it by updating the weights in our VAE based on how 
                    # they contributed to the loss
                # this is how we will do it
                self.train_op = self.optimizer.apply_gradients(grads_and_vars = grads, global_step = self.global_step, name = 'Train_step')
                    # grads_and_vars: A list of (gradient, variable) pairs, where:
                        # gradient is the computed gradient for the variable.
                        # variable is the corresponding trainable variable.
                    # global_step : the global step
            # Training done

            # Now we want to init all of the global vars for the VAE 
            self.init = tf.global_variables_initializer()

################ This is not the Full world model, this is just the CNN-VAE portion, we still need to implement the MDN-RNN