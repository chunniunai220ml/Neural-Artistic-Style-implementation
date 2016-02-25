import numpy as np
import math
from VGGNet import VGGNet

def convolution(inputM, coefficients,bias):
    output = np.zeros( (inputM.shape[0], inputM.shape[1], coefficients.shape[3]) )
    inputs = np.zeros( (inputM.shape[0] + 2, inputM.shape[1] + 2, inputM.shape[2]) )
    inputs[ 1:inputM.shape[0]+1, 1:inputM.shape[0]+1, :] = inputM
    for k in range(coefficients.shape[3]):
        weights = coefficients[:, :, :, k]
        for i in range(inputM.shape[0]):
            for j in range(inputM.shape[1]):
                output[i, j, k] = np.sum( weights *  inputs[i:i+3, j:j+3,:])
    return output + bias
def backpropagation(DeltaI, coefficients):
    sideLength = DeltaI.shape[1]
    prop = np.zeros( (sideLength, sideLength,coefficients.shape[2] ))
    newImage = np.zeros((sideLength + 2, sideLength + 2, DeltaI.shape[2]))
    newImage[1:sideLength + 1, 1:sideLength + 1,:] = DeltaI
    for l in range(coefficients.shape[2]):
        for k in range(DeltaI.shape[2]):
            for i in range(sideLength):
                for j in range(sideLength):
                    prop[i,j,l] += np.sum( newImage[i:i+3, j:j+3, k] * np.rot90(coefficients[:,:,l,k],2)  )
    return prop

# Different implementation using matrix multiplication
def newConvolution(inputM, coefficients, bias):
    totalSize = coefficients.shape[0]*coefficients.shape[1]*coefficients.shape[2]
    x = np.zeros((coefficients.shape[3], totalSize))
    for i in range(coefficients.shape[3]):
        x[i,:] = np.reshape(coefficients[:,:,:,i],(1,totalSize))

    y = np.zeros((totalSize,inputM.shape[0] ** 2))
    newImage = np.zeros((inputM.shape[0] + 2, inputM.shape[1] + 2,inputM.shape[2]))
    newImage[1:inputM.shape[0] + 1, 1:inputM.shape[0]+ 1,:] = inputM
    for i in range(inputM.shape[0]):
        for j in range(inputM.shape[1]):
            y[:,i*inputM.shape[0]+j] = np.reshape(newImage[i:i+3, j:j+3,:], (totalSize))

    z = np.dot(x,y)
    c = np.zeros((inputM.shape[0],inputM.shape[1],coefficients.shape[3]))
    for i in range(coefficients.shape[3]):
        c[:,:,i] = np.reshape(z[i,:],(inputM.shape[0], inputM.shape[0]))
    return c + bias

def newBackProp(inputM,coefficients):
    prop = np.zeros((inputM.shape[0], inputM.shape[0], coefficients.shape[2]))
    totalSize = coefficients.shape[0]*coefficients.shape[1]
    for layer in range(coefficients.shape[2]):
        for m in range(inputM.shape[2]):
            newSlice = np.zeros((inputM.shape[0] + 2, inputM.shape[1] + 2))
            newSlice[1:inputM.shape[0] + 1, 1:inputM.shape[0] + 1] = inputM[:,:,m]
            ASlice = np.zeros((totalSize, inputM.shape[0] ** 2))
            for i in range(inputM.shape[0]):
                for j in range(inputM.shape[1]):
                    ASlice[:,i*inputM.shape[0]+j] = np.ravel(newSlice[i:i+3, j:j+3])
            weights = np.array(np.ravel(np.rot90(coefficients[:,:,layer,m],2)))
            prop[:,:,layer] += np.reshape(np.dot(weights,ASlice),(inputM.shape[0],inputM.shape[1]))
    return prop


if __name__ == '__main__':
    config = {
        "layers" : ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv', 'conv', 'conv', 'pool', 
        'conv', 'conv', 'conv', 'conv', 'pool', 'conv'],

        "style_weights" : [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],

        "content_weights": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

        "pixel_mean" : [103.939, 116.779, 123.680]

    }

    numPool = 4
    for i in range(17):
        j = 16 - i 
        print "layer " + str(j) + ":"
        if config["layers"][j] == "conv":
            print config["style_weights"][j] 
            if config["style_weights"][j] > 0:
                print "layer " + str(j - numPool + 1) + " is taken into content reconstruction"
            if config['content_weights'][j] > 0:
                print "layer " + str(j - numPool + 1) + " is taken into content reconstruction"
        elif config["layers"][j] == "pool":
            numPool =  numPool- 1
        print "============================================="



