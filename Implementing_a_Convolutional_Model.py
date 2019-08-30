import numpy as np
import h5py
import matplotlib.pyplot as plt

def zero_pad(X,pad):
    '''
    Applies a padding of value 'pad' to the numpy array X
    
    Arguments:
        
    X : A numpy array of dimensions: (m,n_C,n_H,n_W) where m is the number of examples, n_C is the depth of
    the channel, n_H is the height of the array and n_W is width of the channel. 
    
    pad: The number of padding layers that need to be applied to the array X
        
    Returns:
    
    X_pad: This is the padded array of dimensions (m,n,n_H+2*pad, n_W+2*pad) 
        
    '''
    X_pad= np.pad(X,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
    
    return X_pad


def conv_single_step(A_prev_slice,W,b):
    '''
    Applies convolution to a single slice from the previous layer
    
    Arguments:
    
    A_prev_slice: A slice of values of dimensions (m,n_C,f,f) from the previous layer where f is the size of 
    the convolution window
    W:Convolution window of dimensions (n_C,f,f)
    b:Bias term of dimensions (1,1,1)
        
    Returns:
    
    Z, a linear transformation of A_prev_slice 
    Z=W*A_prev_slice+b
    
    '''
    Z=np.sum(np.multiply(W,A_prev_slice)) + float(b)
    
    return Z

def conv_forward(A_prev,W,b,hparameters):
    '''
    
    Computes the convolution for a layer and stores the result and the cache for further calculations
    
    Arguments:
        
    A_prev: The nump array of dimensions (m,n_C_prev,n_H,n_W) containing all the activations for all the examples from the previous layer
    W: Weights array of dimensions (n_C,n_C_prev,f,f) 
    b: Bias array of dimensions (1,1,1)
    hparameters: A dictionary that contains the values of padding and stride 
        
    Returns:
    
    Z: The resulting array of convolving the previous layer's array with the weight matrix
    cache: The cache containing A_prev,W,b and hparameters to be used for back propagation
    
    '''
    
    #Get the values of the number of examples, depth of channel, height of the array and the width of the 
    # array from the previous layer and store these values into respective variables
    
    (m,n_C_prev,n_H_prev,n_W_prev)=A_prev.shape[0],A_prev.shape[1],A_prev.shape[2],A_prev.shape[3]
    
    # Get the values of stride and pad from the hparameters dictionary
    pad=hparameters['pad']
    stride=hparameters['stride']
    
    # Get the values of the window size and the depth of the channel for the next layer from the Weight matrix
    
    (n_C,n_C_prev,f,f)=W.shape[0],W.shape[1],W.shape[2],W.shape[3]
    
    # Initialize the results (next layer) array by defining its size and initial values
    n_H=int((n_H_prev+2*pad-f)/stride)+1
    n_W=int((n_W_prev+2*pad-f)/stride)+1
    
    Z= np.zeros((m,n_H,n_W,n_C)) # Z is initialized to an all zero array
    
    
    #Apply padding to the previous layer
    
    A_prev_pad=zero_pad(A_prev,pad)
    
    #Start computation of the forward convolution
    
    for i in range(m): # Looping over all the m exanples
        for n_layer in range (n_C):
            for n_vert in range(n_H):
                for n_hori in range (n_W):
                    A_prev_slice=A_prev_pad[i,:,n_vert*stride:n_vert*stride+f,n_hori*stride:n_hori*stride+f]
                    #Z[i,n_layer,n_vert,n_hori]=np.sum(np.multiply(A_prev_slice,W[n_layer,:,:,:]))+float(b)
                    Z[i,n_layer,n_vert,n_hori]=conv_single_step(A_prev_slice,W[n_layer,:,:,:],b)
                    
    # Assert that the shape of Z is (m,n_C,n_H,n_W)
    assert(Z.shape == (m,n_C,n_H,n_W))
    
    #Cache information for back-propagation
    cache=(A_prev,W,b,hparameters)
    
    return Z,cache
                    
def pool_forward(A_prev,hparameters,mode="max"):
    
    '''
    Implementation of the max pooling function
    
    Arguments:
    A_prev : The layer emerging after the convolution operation (m,n_C_prev,n_H,n_W)
    hparameters: Dictionary containing the dimension f of the pooling window and the stride value
    {'f':x,'stride':y}
    mode: whether max or average pooling needs to be implemented

    Returns:
    Z_pool:The pooled matrix of dimensions [(n-f)/s] + 1 
    

    '''
    # Get the values of the parameters from A_prev array
    (m,n_C_prev,n_H_prev,n_W_prev) = A_prev.shape[0],A_prev.shape[1],A_prev.shape[2],A_prev.shape[3]
    
    #Get the values of stride and f from the hyperparameter dictionary
    
    f=hparameters['f']
    stride=hparameters['stride']
    
    #Define the size of the pooled layer
    
    n_H = int((n_H_prev-f)/stride)+1
    n_W = int((n_W_prev-f)/stride)+1
    n_C= n_C_prev
    
    # Initialize the values of the pooled layer to zeroes
    
    Z_pooled = np.zeros((m,n_C,n_H,n_W))
    
    # Compute the pooling values
    
    for i in range(m):
        for n_layer in range (n_C):
            for n_vert in range (n_H):
                for n_hori in range (n_W):
                    Z_pooled[i,n_layer,n_vert,n_hori]= np.max (A_prev[i,n_layer,n_vert*stride:n_vert*stride+f,n_hori*stride:n_hori*stride+f])
                    
                
    # Assert that the shape of Z is (m,n_C,n_H,n_W)
    assert(Z_pooled.shape == (m,n_C,n_H,n_W))
    
    # Cache values for back-prop
    cache=(A_prev,hparameters)
    
    return Z_pooled,cache

 
    
    
np.random.seed(1)

#Testing the zero_pad function
    
print("Testing the zero_pad function:")

X=np.random.randint(5,size=(2,2,4,4)) # (number_of_examples,number of channels, height of the array, width of the array)
print(X)
X_pad=zero_pad(X,1)
print (X_pad)

print("The first layer of the first example:")
print(X[0,0,:,:])
print("The second layer of the first example:")
print(X[0,1,:,:])

# Testing the conv_single_step_function

print("Testing the conv_single_step_function:") 

A_prev_slice=X[0,:,0:2,0:2] # Take the first example (index 0), take all the channels (:), take height as 2(0:2), take the width as 2(0:2)
W=np.random.randint(5,size=(3,2,2,2))
b=1


print("Input array:")
print(X)

print ('\n'*10)

print("A slice of the input array:")
print(A_prev_slice)
print ("Weight array:")
print(W)
print("Bias term:")
print(b)

Z=conv_single_step(A_prev_slice,W,b)
print("Value after convolution:")
print(Z)


#Testing the conv_forward function

print ("X array:")
print ("\n"*2)
print(X)

print ("W array:")
print ("\n"*2)
print(W)

print ("b array:")
print ("\n"*2)
print(b)



Z,cache=conv_forward(X,W,b,{'pad':1,'stride':2})

print ("\n"*5)

print("Value of the convolved layer:",Z)

print ("\n"*5)
print ("Value of the cache:",cache)

# Testing the pool forward function

print ("X array:")
print ("\n"*2)
print(X)


Z_pooled,cache= pool_forward(X,{'stride':2,'f':2},"max")

print ("\n"*5)

print("Value of the pooled layer:",Z_pooled)


    