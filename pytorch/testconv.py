import torch
import torch.autograd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

in1 = np.array( [ [ 0.0,  50, 0,  29 ],
    [ 0,  80, 31, 2 ],
    [ 33, 90, 0,  75 ],
    [ 0,  9,  0,  95 ] ] )

f1 = np.array( [ [ -1.0, 0, 1 ],
    [ -2, 0, 2 ],
    [ -1, 0, 1 ] ] )

f2 = np.array( [ [ 1.0,  2,  1 ],
    [ 0,  0,  0 ],
    [ -1, -2, 1 ] ] )

input = np.array( [ in1, in1 ] )
filters = np.array( [ f1, f2 ] )
biases = np.array( [ 1 ] )

X = torch.tensor( input.reshape( ( 1, 2, 4, 4 ) ), requires_grad=True, dtype=torch.double )
weight = torch.tensor( filters.reshape( ( 1, 2, 3, 3 ) ), requires_grad=True, dtype=torch.double )
b = torch.tensor( biases, requires_grad=True, dtype=torch.double )

optimizer = optim.SGD( [ X, weight, b ], lr=0.1)
optimizer.zero_grad()

print( "X", X )
print( "weight", weight )
print( "b", b )

Y = torch.nn.functional.conv2d( X, weight, b )

print( "Y", Y )

dY = np.array( [ 0.1, 0.2, 0.3, 0.4 ] ) #, 0.5, 0.6, 0.7, 0.8 ] )

dY = torch.tensor( dY.reshape( 1, 1, 2, 2 ), requires_grad=True, dtype=torch.double )

print( "dY", dY )

Y.backward( dY )
optimizer.step()

print( "Y", Y )

print( "b.grad", b.grad )

print( "weight.grad", weight.grad )

print( "X.grad", X.grad )

print( "X", X )
print( "weight", weight )
print( "b", b )

