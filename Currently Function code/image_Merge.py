import numpy as np
import math
from scipy.misc import imread, imsave, imresize
import time
import Image
from VGGNet import VGGNet

"""
	default learning rate is 2.0 
	default iteration is 500
	default style - content weight ratio is 
		5 : 100
"""


def backPropAct(x, y):
    if y > 0:
        return x
    elif y <= 0:
        return 0

def FFActivation(x):
    if x>0:
        return x
    else:
        return 0

def getGrad(ConvNet, img):
    ReLU = np.vectorize(FFActivation)
    backPropElimZero = np.vectorize(backPropAct)
    ConvNet.F[0] = ReLU(ConvNet.unroll(ConvNet.convolution(img, ConvNet.Conv_W[0], ConvNet.bias[0])))
    
    error = backPropElimZero(ConvNet.F[0] - ConvNet.P[0], ConvNet.F[0])
    return ConvNet.backpropagation(error, ConvNet.Conv_W[0])
    # sideLength = math.sqrt(error.shape[1])
    # error = reshapeBackToGrid(error, sideLength)
    # prop = np.zeros( (sideLength, sideLength, 3))
    # newImage = np.zeros((sideLength + 2, sideLength + 2, error.shape[2]))
    # newImage[1:sideLength + 1, 1:sideLength + 1,:] = error
    # for l in range(ConvNet.Conv_W[0].shape[2]):
    #     for k in range(error.shape[2]):
    #         for i in range(sideLength):
    #             for j in range(sideLength):
    #                 prop[i,j,l] += np.sum( newImage[i:i+3, j:j+3, k] * np.rot90(ConvNet.Conv_W[0][:,:,l,k],2)  )
    # return prop

def ADAM(ConvNet, maxIteration, alpha, img, pixel_mean):
    t = 0
    m = np.zeros((224,224,3))
    v = np.zeros((224,224,3))
    mHat = 0
    vHat = 0
    beta = np.random.random(2)
    sqr = np.vectorize(math.sqrt)
    while t < maxIteration:
        iterationStart = time.time()
        grad = getGrad(ConvNet, img)
        t += 1
        m = beta[0] * m + (1 - beta[0]) * grad
        v = beta[1] * v + (1 - beta[1]) * (grad ** 2)
        mHat = m / (1 - (beta[0] ** t))
        vHat = v / (1 - (beta[1] ** t))
        alphaT = alpha * math.sqrt(1 - beta[1] ** t) / (1 - beta[0] ** t)
        img = img - alphaT * mHat / (sqr(vHat) + 1e-20)

        #saving progress
        np.savetxt("result/testIntegrate/whiteImageLayer1.txt", img[:, :, 0] + pixel_mean[0])
        np.savetxt("result/testIntegrate/whiteImageLayer2.txt", img[:, :, 1] + pixel_mean[1])
        np.savetxt("result/testIntegrate/whiteImageLayer3.txt", img[:, :, 2] + pixel_mean[2])
        img[:,:,0] =  img[:, :, 0] + pixel_mean[0]
        img[:,:,1] =  img[:, :, 1] + pixel_mean[1]
        img[:,:,2] =  img[:, :, 2] + pixel_mean[2]
        result = Image.fromarray((img).astype(np.uint8))
        result.save("result/testIntegrate/content" + str(t) +".bmp")
        iterationEnd = time.time()        
        img[:,:,0] =  img[:, :, 0] - pixel_mean[0]
        img[:,:,1] =  img[:, :, 1] - pixel_mean[1]
        img[:,:,2] =  img[:, :, 2] - pixel_mean[2]

        print "Complete iteration: " + str(t) + "   It took " + str(iterationEnd - iterationStart) + " seconds"
    return img

def run():
    ConvNet = VGGNet()
    ReLU = np.vectorize(FFActivation)

    #Preprocessing Art and Photo
    pixel_mean = [103.939, 116.779, 123.680]
    art = imread('result/starryNight.jpg')
    art = imresize(art, (224, 224))
    art[:, :, 0] = art[:, :, 0] - pixel_mean[0]
    art[:, :, 1] = art[:, :, 1] - pixel_mean[1]
    art[:, :, 2] = art[:, :, 2] - pixel_mean[2]

    photo = imread('result/stanford.jpeg')
    photo = imresize(photo, (224, 224))
    photo[:,:,0] = photo[:, :, 0] - pixel_mean[0]
    photo[:,:,1] = photo[:, :, 1] - pixel_mean[1]
    photo[:,:,2] = photo[:, :, 2] - pixel_mean[2]
    print "load image complete"

    whiteImage = np.ones((224, 224, 3)) * 255
    whiteImage[:,:,0] -= pixel_mean[0]
    whiteImage[:,:,1] -= pixel_mean[1]
    whiteImage[:,:,2] -= pixel_mean[2]


    configuration = {
        "layers" : ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv', 'conv', 'conv', 'pool', 
        'conv', 'conv', 'conv', 'conv', 'pool', 'conv'],

        "style_weights" : [1/5, 0, 0, 1/5, 0, 0, 1/5, 0, 0, 0, 0, 1/5, 0, 0, 0, 0, 1/5],

        "content_weights": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],

        "pixel_mean" : pixel_mean
    }

    #initial passing of Photo through ConvNet
    for i in range(13):
        ConvNet.F[i] = np.load("Data/F"+str(i)+".npy")
        ConvNet.A[i] = np.load("Data/A"+str(i)+".npy")

    #initial passing of Photo through ConvNet
    # output = photo
    # for i in range(13):
    #     convolvedImage = ConvNet.convolution(output, ConvNet.Conv_W[i],ConvNet.bias[i])
    #     output = ReLU( convolvedImage )
    #     ConvNet.P[i] = ConvNet.unroll(output)
    #     np.save("Data/F"+ str(i)+".npy", ConvNet.P[i])
    #     if i in [1, 3, 7, 11]:
    #         output = ConvNet.pool(output)

    # output = art
    # for i in range(13):
    #     convolvedImage = ConvNet.convolution(output, ConvNet.Conv_W[i],ConvNet.bias[i])
    #     output = ReLU( convolvedImage )
    #     ConvNet.F[i] = ConvNet.unroll(output)
    #     ConvNet.A[i] = np.dot(ConvNet.F[i], ConvNet.F[i].T)
    #     np.save("Data/A"+ str(i)+".npy", ConvNet.A[i])
    #     if i in [1, 3, 7, 11]:
    #         output = ConvNet.pool(output)

    print "initial feed forward complete"

    #run optimization
    ADAM(ConvNet, 500, 2, whiteImage, pixel_mean)
if __name__ == '__main__':
    run()