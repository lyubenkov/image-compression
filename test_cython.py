# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rawpy
import math
import scipy.linalg as la
from multiprocessing.dummy import Pool as ThreadPool
import functools
from PIL import Image

# The function returns a diagonal matrix from the given M block, repeating it N times
def diagM(M,n):
    return la.block_diag(*([M]*n))

def rgb2grey(rgb):
    return np.dot(rgb[:,:,:3], [.299, .587, .114])


"""
Bandpass filter
n - filter matrix size
w - cutoff frequency squared
"""
def filter_stripe(n,w):
    F = np.zeros((n,n))
    w = int(round(math.sqrt(w)))
    for i in xrange(0, w):
        for j in xrange(0, w):
            F[i][j]=1;

    return F;

'''
Hyperbolic filter
s - size
k - cutoff frequency squared
r - rounding order
'''
def filter_hyperbole(s,k,r):
    F = np.ones((s,s))
    for i in xrange(0, s):
        for j in xrange(0, s):
            F[i][j] = (float(1)/(1+(i+1)*(j+1)/k))
            F=F.round(r)

    return F;

'''
Butterworth filter
s - filter matrix size
w - cutoff frequency squared
m - filter order
r - rounding order
'''
def filter_butterworth(s,w,n,r):
    F = np.ones((s,s))
    for i in xrange(0, s):
        for j in xrange(0, s):
            F[i][j] = float(1)/(1+(((i+1)*(j+1)/w) ** (2*n)));
            F=F.round(r)

    return F;


'''
multm - multiplication of a square matrix by a required number of times
'''
def multm(F,m):
    s = F.shape[0]
    FM = np.zeros((s*m,s*m))
    for ii in xrange(0, m):
        for jj in xrange(0, m):
            for i in xrange(0, s):
                for j in xrange(0, s):
                    FM[i+ii*s][j+jj*s]=F[i][j]

    return FM

def loadm(path):
    f = open ( path , 'r')
    l = np.array([ map(float,line.split(',')) for line in f if line.strip() != "" ])

    return l

'''
DCT matrix
To check: np.dot(dct(30), np.transpose(dct(30)))
'''
def dct(n):
    F = np.zeros((n,n))
    for i in xrange(0, n):
        F[i][0]=1/math.sqrt(n)
    for i in xrange(0, n):
        for j in xrange(1, n):
            F[i][j] = math.sqrt(float(2)/n)*math.cos((2*i+1)*j*math.pi/(2*n))

    return F


def m4img(P,M):
    ip = np.asarray(P.shape) // np.asarray(M.shape)
    mp = np.asarray(P.shape) % np.asarray(M.shape)
    tp = np.asarray(M.shape) - mp
    NM = diagM(M, np.asarray(M.shape) * (ip + 1)) # generating a matrix with a 1 size M greater than the image
    NM = NM[0:-tp, 0:-tp] # cropping the excess     if mp.any() != 0 :

    return NM


def createFP_multi(i,j):
    global P, FP
    if (i < P.shape[0] and j < P.shape[1]):
        FP[i][j] = P[i][j]
    else:
        if i < P.shape[0]:
            FP[i][j] = P[i][P.shape[1] - 1]
        else:
            FP[i][j] = FP[P.shape[0] - 1][j]


def createFP(M):
    global P, FP
    mp = np.asarray(P.shape) % np.asarray(M.shape)
    if mp[1] != 0 :
        n = P.shape[1] - mp[1] + M.shape[1]
    else:
        n = P.shape[1]
    FP = np.zeros((n,n))
    for i in xrange(0, n):
        for j in xrange(0, n):
            if (i < P.shape[0] and j < P.shape[1]):
                FP[i][j] = P[i][j]
            else:
                if i < P.shape[0]:
                    FP[i][j] = P[i][P.shape[1] - 1]
                else:
                    FP[i][j] = FP[P.shape[0] - 1][j]

'''
    for i in xrange(0, n):
        for j in xrange(0, n):
            if (i < P.shape[0] and j < P.shape[1]):
                FP[i][j] = P[i][j]
            else:
                if i < P.shape[0]:
                    FP[i][j] = FP[i][j - 1]
                else:
                    FP[i][j] = FP[i - 1][j]
'''


'''
    // PREPARING MATRICES, CALCULATING SPECTRUM
    {{P='R.xml'; P=P*C-C/2; M='HW16.xml'}}
    M= orth(M);
    M=norms(M);
    P=rowcol(P,0,63,0,63);
    D=diag(M,4); {{S=D'*P*D}};
    
    //INITIALIZING THE FILTER
    F=filter(16,4,6);

    // APPLYING THE FILTER
    {{SF=S.*F}}
    SF=roundm(SF);

    // RESTORING THE IMAGE
    {{P=D*SF*D'; P=P+C/2;P=P/C}}
    putm(P);
    plots(P,'XP');
'''

if __name__ == "__main__":
#    image = mpimg.imread('./resources/DSC_0214.tif')
#    raw = rawpy.imread('./resources/DSC_0214.NEF')
    raw = rawpy.imread('./resources/flowers-61.nef')
    image = raw.postprocess(use_camera_wb = True, output_bps=8)
    P = rgb2grey(image)
    Image.fromarray(P).convert('RGB').save('./output/test.jpg', 'JPEG', quality=80)

    # Setting the bit depth
    C=1000
    FP = np.zeros((1,1))

    M = loadm('./matrix/HW32')
    createFP(M)

    FP=FP*C-C/2
    FM = diagM(M, FP.shape[0] / M.shape[0])
    F = multm(filter_butterworth(M.shape[0], 100, 3, 3),FP.shape[0] / M.shape[0])

    S = np.dot(np.transpose(FM), FP)
    S = np.dot(FP, FM)

    SF = S*F
    SF = SF[0:(P.shape[0] - 1), 0:(P.shape[1] - 1)]

    rescaled = (255.0 / SF.max() * (SF - SF.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('./output/test.png')

    plt.subplot(1,4,1)
    plt.xticks([]),plt.yticks([])
    plt.title("Original")
    plt.imshow(image)

    plt.subplot(1,4,2)
    plt.xticks([]),plt.yticks([])
    plt.title("Weighted Average Square")
    plt.imshow(FP, cmap = cm.Greys_r)

    plt.subplot(1,4,3)
    plt.xticks([]),plt.yticks([])
    plt.title("Matrix")
    plt.imshow(FM, cmap = cm.Greys_r,interpolation='nearest')

    plt.subplot(1,4,4)
    plt.xticks([]),plt.yticks([])
    plt.title("SF")
    plt.imshow(SF, cmap = cm.Greys_r)
#    plt.imshow(filter_stripe(100, 99), cmap = cm.Greys_r, interpolation='nearest')
#    plt.imshow(filter_hyperbole(100, 99, 3), cmap = cm.Greys_r, interpolation='nearest')
#    plt.imshow(multm(filter_butterworth(100, 200, 3, 3),10), cmap = cm.Greys_r)
#    plt.imshow(loadm('./matrix/HW32'), cmap = cm.Greys_r, interpolation='nearest')
#    plt.imshow(dct(30), cmap = cm.Greys_r, interpolation='nearest')
#    plt.imshow(FM, cmap = cm.Greys_r, interpolation='nearest')
    plt.show()


'''
    plt.subplot(2,2,2)
    plt.xticks([]),plt.yticks([])
    plt.title("Average")
    plt.imshow(grey, cmap = cm.Greys_r)
'''

'''
    plt.axis("off")
    plt.imshow(grey, cmap = cm.Greys_r, interpolation='nearest')
    plt.show()
'''

'''
import matplotlib.image as mpimg

import tifffile as tiff
'''