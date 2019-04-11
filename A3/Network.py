# Network.py
# (C) Jeff Orchard, 2019

import numpy as np


#============================================================
#
# Untility functions
#
#============================================================
def NSamples(x):
    '''
        n = NSamples(x)
        
        Returns the number of samples in a batch of inputs.
        
        Input:
         x   is a 2D array
        
        Output:
         n   is an integer
    '''
    return len(x)

def Shuffle(inputs, targets):
    '''
        s_inputs, s_targets = Shuffle(inputs, targets)
        
        Randomly shuffles the dataset.
        
        Inputs:
         inputs     array of inputs
         targets    array of corresponding targets
         
        Outputs:
         s_inputs   shuffled array of inputs
         s_targets  corresponding shuffled array of targets
    '''
    data = list(zip(inputs,targets))
    np.random.shuffle(data)
    s_inputs, s_targets = zip(*data)
    return np.array(s_inputs), np.array(s_targets)

def MakeBatches(data_in, data_out, batch_size=10, shuffle=True):
    '''
    batches = MakeBatches(data_in, data_out, batch_size=10)
    
    Breaks up the dataset into batches of size batch_size.
    
    Inputs:
      data_in    is a list of inputs
      data_out   is a list of outputs
      batch_size is the number of samples in each batch
      shuffle    shuffle samples first (True)
      
    Output:
      batches is a list containing batches, where each batch is:
                 [in_batch, out_batch]
    
    Note: The last batch might be incomplete (smaller than batch_size).
    '''
    N = len(data_in)
    r = range(N)
    if shuffle:
        r = np.random.permutation(N)
    batches = []
    for k in range(0, N, batch_size):
        if k+batch_size<=N:
            din = data_in[r[k:k+batch_size]]
            dout = data_out[r[k:k+batch_size]]
        else:
            din = data_in[r[k:]]
            dout = data_out[r[k:]]
        if isinstance(din, (list, tuple)):
            batches.append( [np.stack(din, dim=0) , np.stack(dout, dim=0)] )
        else:
            batches.append( [din , dout] )
  
    return batches


# Cost Functions--------------------------
def CrossEntropy(y, t):
    '''
        E = CrossEntropy(y, t)

        Evaluates the mean cross entropy loss between outputs y and targets t.

        Inputs:
          y is a batch of outputs
          t is a batch of targets

        Outputs:
          E is the average cross entropy for the batch
    '''
    # Cross entropy formula
    E = -np.sum( t*np.log(y) + (1.-t)*np.log(1.-y) )
    return E / NSamples(t)

def CrossEntropy_p(y, t):
    return (y-t) / y / (1.-y) / NSamples(y)

def MSE(y, t):
    '''
        E = MSE(y, t)

        Evaluates the MSE loss function between outputs y and targets t.

        Inputs:
          y is a batch of outputs
          t is a batch of targets

        Outputs:
          E is the loss function for the batch
    '''
    # [1] MSE formula
    E = np.sum((y-t)**2)/NSamples(t)
    return E

def MSE_p(y, t):
    return (y-t)/ NSamples(t)

def CategoricalCE(outputs, targets):
    return -np.sum(targets * np.log(outputs)) / NSamples(targets)

def ClassificationAccuracy(outputs, targets):
    '''
        a = ClassificationAccuracy(outputs, targets)
        
        Returns the fraction (between 0 and 1) of correct classifications
        in the dataset. The predicted class is taken to be the one-hot of the outputs.

        Inputs:
          outputs is a batch of output vectors
          targets is a batch of target classification vectors

        Outputs:
          a is a number in (0,1) giving the fraction of correct classifications
    '''
    yb = OneHot(outputs)
    n_incorrect = np.sum(yb!=targets) / 2.
    return 1. - float(n_incorrect) / NSamples(outputs)


# Activation Functions--------------------------
def Logistic(z):
    '''
        y = Logistic(z)

        Applies the logistic function to each element in z.

        Input:
         z    is a scalar, list or array

        Output:
         y    is the same shape as z
    '''
    return 1. / (1 + np.exp(-z) )

def Logistic_p(h):
    '''
        yp = Logistic_p(h)
        
        Returns the slope of the logistic function at z when h = Logistic(z).
        Note that h (node activity) is the input, NOT z (input current).
    '''
    return h*(1.-h)

def Logistic_z_p(z):
    '''
        yp = Logistic_z_p(z)
        
        Returns the slope of the logistic function at z when h = Logistic(z).
        Note that z (input current) is the input, NOT h (node activity).
    '''
    return Logistic(z)*(1.-Logistic(z))

def Tanh(z):
    return np.tanh(z)

def Tanh_p(h):
    return 1. - h**2

def Tanh_z_p(z):
    return 1. - np.tanh(z)**2

def Softmax(z):
    v = np.exp(z)
    s = np.sum(v, axis=1)
    return v/np.tile(s[:,np.newaxis], [1,np.shape(v)[1]])

def ReLU(z):
    return np.clip(z, 0., None)

def ReLU_p(h):
    return np.clip(np.sign(h), 0, 1)

def Arctan(z):
    return np.arctan(z)

def Arctan_z_p(z):
    return 1. / (1.+z**2)

def Identity(z):
    '''
        y = Identity(z)

        Does nothing... simply returns z.

        Input:
         z    is a scalar, list or array

        Output:
         y    is the same shape as z
    '''
    return z

def Identity_p(h):
    '''
        yp = Identity_p(h)
        
        Returns the slope of the identity function h.
    '''
    return np.ones_like(h)

def OneHot(z):
    '''
        y = OneHot(z)

        Applies the one-hot function to the vectors in z.
        Example:
          OneHot([[0.9, 0.1], [-0.5, 0.1]])
          returns np.array([[1,0],[0,1]])

        Input:
         z    is a 2D array of samples

        Output:
         y    is an array the same shape as z
    '''
    y = []
    # Locate the max of each row
    for zz in z:
        idx = np.argmax(zz)
        b = np.zeros_like(zz)
        b[idx] = 1.
        y.append(b)
    y = np.array(y)
    return y



#==================================================
#
# Layer Class
#
#==================================================
class Layer():
    
    def __init__(self, n_nodes, act=None):
        '''
            lyr = Layer(n_nodes, act='logistic')
            
            Creates a layer object.
            
            Inputs:
             n_nodes  the number of nodes in the layer
             act      specifies the activation function
                      Use 'logistic' or 'identity'
        '''
        self.N = n_nodes  # number of nodes in this layer
        self.z = []       # node input currents
        self.h = []       # node activities
        self.mask = []    # mask for non-dropped nodes
        self.b = np.zeros(self.N)  # biases
        
        # Activation functions
        if act=='logistic':
            self.act_text = 'logistic'
            self.sigma = Logistic
            self.sigma_p = Logistic_p
            self.sigma_z_p = Logistic_z_p
        elif act=='ReLU':
            self.act_text = 'ReLU'
            self.sigma = ReLU
            self.sigma_p = ReLU_p
            self.sigma_z_p = ReLU_p
        elif act=='arctan':
            self.act_text = 'arctan'
            self.sigma = Arctan
            #self.sigma_p = Arctan_p
            self.sigma_z_p = Arctan_z_p
        elif act=='tanh':
            self.act_text = 'tanh'
            self.sigma = Tanh
            self.sigma_p = Tanh_p
            self.sigma_z_p = Tanh_z_p
        elif act=='softmax':
            self.act_text = 'softmax'
            self.sigma = Softmax
        else:
            self.act_text = 'identity'
            self.sigma = Identity
            self.sigma_p = Identity_p
            self.sigma_z_p = Identity_p



#==================================================
#
# Network Class
#
#==================================================
class Network():

    def __init__(self, cost='cross-entropy'):
        '''
            net = Network(cost='cross-entropy')

            Creates an empty Network object.

            Inputs:
              cost is a string indicating the cost function
                   Options include:
                      'cross-entropy',
                      'categorical-cross-entropy',
                      'MSE'
        '''
        self.n_layers = 0
        self.lyr = []    # a list of Layers
        self.W = []      # Weight matrices, indexed by the layer below it
        self.cost = CrossEntropy
        self.cost_p = CrossEntropy_p
        self.cost_text = 'cross-entropy'
        if cost=='MSE':
            self.cost = MSE
            self.cost_p = MSE_p
            self.cost_text = 'MSE'
        elif cost=='categorical-cross-entropy':
            self.cost = CategoricalCE
            self.cost_p = None #CategoricalCE_p
            self.cost_text = 'categorical-cross-entropy'


    def AddLayer(self, layer):
        '''
            net.AddLayer(layer)

            Adds the layer object to the network and connects it to the preceding layer.

            Inputs:
              layer is a layer object
        '''
        self.lyr.append(layer)
        self.n_layers += 1
        # If this isn't our first layer, add connection weights
        if self.n_layers>=2:
            m = self.lyr[-1].N
            n = self.lyr[-2].N
            temp = np.random.normal(size=[n,m])/np.sqrt(n)*4.
            self.W.append(temp)

    def TopGradient(self, targets):
        '''
            dEdz = net.TopGradient(targets)

            Computes and returns the gradient of the cost with respect to the input current
            to the output nodes.

            Inputs:
              targets is a batch of targets corresponding to the last FeedForward run

            Outputs:
              dEdz is a batch of gradient vectors corresponding to the output nodes
        '''
        if self.cost_text=='categorical-cross-entropy' and self.lyr[-1].act_text=='softmax':
            return ( self.lyr[-1].h - targets ) / NSamples(targets)
        return self.cost_p(self.lyr[-1].h, targets) * self.lyr[-1].sigma_p(self.lyr[-1].h) / NSamples(targets)

    def FeedForward(self, x):
        '''
            y = net.FeedForward(x)

            Runs the network forward, starting with x as input.
            Returns the activity of the output layer.

        '''
        x = np.array(x)  # Convert input to array, in case it's not
        
        self.lyr[0].h = x # Set input layer

        # Loop over connections...
        for pre,post,W in zip(self.lyr[:-1], self.lyr[1:], self.W):

            # Calc. input current to next layer
            post.z = pre.h @ W + post.b
            
            # Use activation function to get activities
            post.h = post.sigma(post.z)
        
        # Return activity of output layer
        return self.lyr[-1].h


    def BackProp(self, t, lrate=0.05):
        '''
            net.BackProp(targets, lrate=0.05)
            
            Given the current network state and targets t, updates the connection
            weights and biases using the backpropagation algorithm.
            
            Inputs:
             t      an array of targets (number of samples must match the
                    network's output)
             lrate  learning rate
        '''
        t = np.array(t)  # convert t to an array, in case it's not
                
        # Error gradient for top layer
        dEdz = self.TopGradient(t) 
        
        # Loop down through the layers
        for i in range(self.n_layers-2, -1, -1):
            pre = self.lyr[i]
            
            # Gradient w.r.t. weights
            dEdW = pre.h.T @ dEdz
            
            # Gradient w.r.t. biases
            dEdb = np.sum(dEdz, axis=0)
            
            # If not the bottom layer,
            # Project error gradient down to layer below
            if i>0:
                dEdz = ( dEdz @ self.W[i].T ) * pre.sigma_z_p(pre.z)
            
            # Update weights and biases
            self.W[i] -= lrate*dEdW
            self.lyr[i+1].b -= lrate*dEdb


    def SGD(self, inputs, targets, lrate=0.05, epochs=1, batch_size=10):
        '''
            progress = net.SGD(inputs, targets, lrate=0.05, epochs=1, batch_size=10)

            Performs Stochastic Gradient Descent on the network.
            Run through the dataset in batches 'epochs' number of times, incrementing the
            network weights after each batch. For each epoch, it
            shuffles the dataset.

            Inputs:
              inputs  is an array of input samples
              targets is a corresponding array of targets
              lrate   is the learning rate (try 0.001 to 5.)
              epochs  is the number of times to go through the dataset
              batch_size is the number of samples per batch
              
            Outputs:
              progress is an (expochs)x2 array with epoch in the first column, and 
                      cost in the second column
        '''
        loss_history = []
        for k in range(epochs):
            batches = MakeBatches(inputs, targets, batch_size=10, shuffle=True)
            for mini_batch in batches:
                self.FeedForward(mini_batch[0])
                self.BackProp(mini_batch[1])

            loss_history.append([k, self.Evaluate(inputs, targets)])

        return np.array(loss_history)


    def Evaluate(self, inputs, targets):
        '''
            E = net.Evaluate(inputs, targets)

            Computes the average loss over the supplied dataset.

            Inputs
             inputs  is a batch of inputs
             targets is a batch of corresponding targets

            Outputs
             E is a scalar, the average loss
        '''
        y = self.FeedForward(inputs)
        return self.cost(y, targets)









# end