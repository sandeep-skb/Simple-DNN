import numpy as np

# This function adds non-linearity to the outputs.
# When deriv=True, it computes derivative of the activation
# which in this case is sigmoid.
def non_lin(x, deriv=False):
	if deriv == True:
		return x*(1-x)
	return 1/(1+np.exp(-x))


#Input data
X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])
X = X.T # dim (3,4)

# Output data
Y = np.array([[0],
			  [1],
			  [1],
			  [0]])
Y = Y.T # dim( 1, 4)

#Learning rate
alpha = 0.5

np.random.seed(1)

W1 = 2*np.random.random((4,3)) - 1
W2 = 2*np.random.random((4,4)) - 1
W3 = 2*np.random.random((1,4)) - 1

for j in range(60000):
    l0 = X
    l1 = non_lin(np.dot(W1, X)) #dim(4,3).dim(3,4)= dim(4,4)
    l2 = non_lin(np.dot(W2, l1)) #dim(4,4).dim(4,4) = dim(4,4)
    l3 = non_lin(np.dot(W3, l2)) #dim(1,4).dim(4,4) = dim(1,4)
    
    # Calculating the loss.
    loss = l3 - Y #dim(1,4)
    # Priting loss after 10k iterations.
    if (j%10000 == 0):
        print ((np.mean(np.abs(loss))))
    l3_error = loss
    l3_delta = l3_error * non_lin(l3, deriv=True) #dim(1,4) * dim(1,4)
    
    l2_error = np.dot(W3.T, l3_delta) #dim(4,1).dim(1,4) = dim(4,4)
    l2_delta = l2_error * non_lin(l2, deriv=True) #dim(4,4) * dim(4,4)
    
    l1_error = np.dot(W2.T, l2_delta) #dim(4,4).dim(4,4) = dim(4,4)
    l1_delta = l1_error*non_lin(l1,deriv=True) #dim(4,4).dim(4,4) = dim(4,4)
    
    W3 -= alpha*np.dot(l3_delta, l2.T) #dim(1,4).dim(4,4) = dim(1,4)
    W2 -= alpha*np.dot(l2_delta, l1.T) #dim(4,4).dim(4,4) = dim(4,4)
    W1 -= alpha*np.dot(l1_delta, l0.T) #dim(4,4).dim(4,3) = dim(4,3)
# Printing final result.	
print (l2)
