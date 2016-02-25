import numpy as np
import math

class VGGNet:

    #4D randomized Weights and bias for each of the 16 convolution layer
    Conv_W = []
    bias = 0

    #For content matching: Nl X Ml 2D matrices/ activation of PHOTOGRAPH
    P = []

    #For style matching: Nl x Nl 2D matrices/ inner product of activation of ART
    A = []

    #For content matching: Nl X Ml 2D matrices/ activation of x
    F = []

    #For style matching: Nl x Nl 2D matrices/ inner product of activation of x
    G = []



    def pool(self, inputM):
        output = np.zeros((inputM.shape[0]/2, inputM.shape[1]/2, inputM.shape[2]))
        for i in range(inputM.shape[0]/2 ):
            for j in range(inputM.shape[1]/2):
                for k in range(inputM.shape[2]):
                    output[i, j, k] = np.sum(inputM[2 * i: 2 * i + 2, 2 * j : 2 * j + 2, k])/4
        return output

    def unroll(self, inputM):
        output = np.zeros((inputM.shape[2], inputM.shape[1] * inputM.shape[0]))
        for i in range(inputM.shape[2]):
            output[i, :] = np.ravel(inputM[:, :, i])
        return output

    def upsample(self, inputM):
        output = np.zeros((inputM.shape[0] * 2, inputM.shape[1] * 2, inputM.shape[2]))
        
        for i in range(inputM.shape[0] * 2 ):
            for j in range(inputM.shape[1] * 2):
                for k in range(inputM.shape[2]):
                    output[i, j, k] = inputM[i/2, j/2, k] / 4
        return output

    def convolution(self, inputM, coefficients,bias):
        output = np.zeros( (inputM.shape[0], inputM.shape[1], coefficients.shape[3]) )
        inputs = np.zeros( (inputM.shape[0] + 2, inputM.shape[1] + 2, inputM.shape[2]) )
        inputs[ 1:inputM.shape[0]+1, 1:inputM.shape[0]+1, :] = inputM
        for k in range(coefficients.shape[3]):
            weights = coefficients[:, :, :, k]
            for i in range(inputM.shape[0]):
                for j in range(inputM.shape[1]):
                    output[i, j, k] = np.sum( weights *  inputs[i:i+3, j:j+3,:])
        return output + bias

    def backpropagation(self, DeltaI, coefficients):
        sideLength = int(math.sqrt(DeltaI.shape[1]))
        D = np.zeros((sideLength, sideLength, DeltaI.shape[0]))
        for k in range(DeltaI.shape[0]):
            for i in range(int(sideLength)):
                for j in range(int(sideLength)):
                    D[i, j, k] = DeltaI[k, sideLength * i + j]
        prop = np.zeros( (sideLength, sideLength, 3))
        newImage = np.zeros((sideLength + 2, sideLength + 2, D.shape[2]))
        newImage[1:sideLength + 1, 1:sideLength + 1,:] = D
        for l in range(coefficients.shape[2]):
            for k in range(D.shape[2]):
                for i in range(sideLength):
                    for j in range(sideLength):
                        prop[i,j,l] += np.sum( newImage[i:i+3, j:j+3, k] * np.rot90(coefficients[:,:,l,k],2)  )
        return prop



    def __init__(self):
        VGGNet.Conv_W.append( np.random.normal(0, 0.1, (3, 3, 3, 64)))
        VGGNet.Conv_W.append( np.random.random((3, 3, 64, 64))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 64, 128))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 128, 128))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 128, 256))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 256, 256))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 256, 256))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 256, 256))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 256, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )
        VGGNet.Conv_W.append( np.random.random((3, 3, 512, 512))/16 - 1/32 )

        VGGNet.bias = np.random.random(16)

        VGGNet.P.append( np.zeros( (64, 224*224) ) )
        VGGNet.P.append( np.zeros( (64, 224*224) ) )
        VGGNet.P.append( np.zeros( (128, 112*112) ) )
        VGGNet.P.append( np.zeros( (128, 112*112) ) )
        VGGNet.P.append( np.zeros( (256, 56*56) ) )
        VGGNet.P.append( np.zeros( (256, 56*56) ) )
        VGGNet.P.append( np.zeros( (256, 56*56) ) )
        VGGNet.P.append( np.zeros( (256, 56*56) ) )
        VGGNet.P.append( np.zeros( (512, 28*28) ) )
        VGGNet.P.append( np.zeros( (512, 28*28) ) )
        VGGNet.P.append( np.zeros( (512, 28*28) ) )
        VGGNet.P.append( np.zeros( (512, 28*28) ) )
        VGGNet.P.append( np.zeros( (512, 14*14) ) )
        VGGNet.P.append( np.zeros( (512, 14*14) ) )
        VGGNet.P.append( np.zeros( (512, 14*14) ) )
        VGGNet.P.append( np.zeros( (512, 14*14) ) )

        VGGNet.F.append( np.zeros( (64, 224*224) ) )
        VGGNet.F.append( np.zeros( (64, 224*224) ) )
        VGGNet.F.append( np.zeros( (128, 112*112) ) )
        VGGNet.F.append( np.zeros( (128, 112*112) ) )
        VGGNet.F.append( np.zeros( (256, 56*56) ) )
        VGGNet.F.append( np.zeros( (256, 56*56) ) )
        VGGNet.F.append( np.zeros( (256, 56*56) ) )
        VGGNet.F.append( np.zeros( (256, 56*56) ) )
        VGGNet.F.append( np.zeros( (512, 28*28) ) )
        VGGNet.F.append( np.zeros( (512, 28*28) ) )
        VGGNet.F.append( np.zeros( (512, 28*28) ) )
        VGGNet.F.append( np.zeros( (512, 28*28) ) )
        VGGNet.F.append( np.zeros( (512, 14*14) ) )
        VGGNet.F.append( np.zeros( (512, 14*14) ) )
        VGGNet.F.append( np.zeros( (512, 14*14) ) )
        VGGNet.F.append( np.zeros( (512, 14*14) ) )

        VGGNet.A.append( np.zeros( (64, 64) ) )
        VGGNet.A.append( np.zeros( (64, 64) ) )
        VGGNet.A.append( np.zeros( (128, 128) ) )
        VGGNet.A.append( np.zeros( (128, 128) ) )
        VGGNet.A.append( np.zeros( (256, 256) ) )
        VGGNet.A.append( np.zeros( (256, 256) ) )
        VGGNet.A.append( np.zeros( (256, 256) ) )
        VGGNet.A.append( np.zeros( (256, 256) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )
        VGGNet.A.append( np.zeros( (512, 512) ) )

        VGGNet.G.append( np.zeros( (64, 64) ) )
        VGGNet.G.append( np.zeros( (64, 64) ) )
        VGGNet.G.append( np.zeros( (128, 128) ) )
        VGGNet.G.append( np.zeros( (128, 128) ) )
        VGGNet.G.append( np.zeros( (256, 256) ) )
        VGGNet.G.append( np.zeros( (256, 256) ) )
        VGGNet.G.append( np.zeros( (256, 256) ) )
        VGGNet.G.append( np.zeros( (256, 256) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
        VGGNet.G.append( np.zeros( (512, 512) ) )
