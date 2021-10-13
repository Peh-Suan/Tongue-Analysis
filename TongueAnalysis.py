import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import glob
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image, ImageDraw
import seaborn as sns
import scipy
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr


class TongueAnalysis:
    def __init__(self, filePath, radii, center, pictureType = 'png'):
        fileL = glob.glob(filePath + '*.' + pictureType)
        (radius1, radius2) = (radii[0], radii[1])
        (xc, yc) = (center[0], center[1])
        img = cv.imread(fileL[0])
        mask1 = np.zeros_like(img)
        mask1 = cv.circle(mask1, (xc,yc), radius1, (255,255,255), -1)
        mask2 = np.zeros_like(img)
        mask2 = cv.circle(mask2, (xc,yc), radius2, (255,255,255), -1)
        mask = cv.subtract(mask2, mask1)
        
        for count, file in enumerate(fileL):
            image = cv.imread(file)
            flatImage = image.reshape(-1)
            flatMask = mask.reshape(-1)
            for i in range(len(flatImage)):
                if flatMask[i] == 0:
                    flatImage[i] = 0
            imgMasked = flatImage.reshape(image.shape)
            cv.imwrite(file.replace('.' + pictureType, '_masked.' + pictureType), imgMasked)
            cv.destroyAllWindows()
            
            if count + 1 != len(fileL):
                print('Masking ultrasound frames: ' + str((count + 1)*100/len(fileL))[:4] + '% completed.', end = '\r')
            else:
                print(' '*50, end = '\r')
                print('', end = '\r')
        
        imgDenoisedL = []
        fileL = glob.glob(filePath + '*_masked.' + pictureType)
        for count, file in enumerate(fileL):
            imgCropped = np.array(Image.open(file).convert('RGBA').crop((216, 80, 896, 490)))
            imgInverted = cv.bitwise_not(imgCropped)
            imgInverted = plt.imshow(imgInverted.mean(2), cmap = 'gray')
            
            imgDenoised = cv.threshold(imgInverted.get_array().data, 170, 600, cv.THRESH_TRUNC)[1]
            
            plt.axis('off')
            plt.close()
            
            imgDenoisedL.append(imgDenoised)
        

            if count + 1 != len(fileL):
                print('Denoising ultrasound frames: ' + str((count + 1)*100/len(fileL))[:4] + '% completed.', end = '\r')
            else:
                print(' '*50, end = '\r')
                print('', end = '\r')
                
        imgAll = np.stack(imgDenoisedL)
        
        pca = PCA(n_components = imgAll.shape[0], random_state = 9527)
        
        
        imgFlattened = np.reshape(imgAll, (imgAll.shape[0], -1))
        imgTransformed = pca.fit_transform(imgFlattened)
        
        cumExplained = plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Principal components')
        plt.ylabel('Cumulative explained variance')
        self.cumulativeExplainedVarianceRatio = cumExplained[0]
        plt.close()
        
        for file in glob.glob(filePath + '*_masked.' + pictureType):
            os.remove(file)
        
        self.images = imgAll
        
        self.pca = pca
        
        self.PC = self.pca.components_
        
        self.imgTransformed = pca.transform(imgFlattened)
        
        self.imgReconstructed = minmax_scale(pca.inverse_transform(imgTransformed), axis=1)
               
    def plot_PC(self, PCNum, cmap = 'gray', figsize = (20, 20), printTitle = False):
        shape = (self.images.shape[1], self.images.shape[2])
        fig, axes = plt.subplots(figsize = figsize)
        
        if printTitle:
            axes.title.set_text(f'PC{PCNum}')
        axes.axis('off')
        axes.imshow(minmax_scale(self.pca.components_, axis = 1)[PCNum - 1].reshape(shape), cmap = cmap)
        
    def plot_reconstructed_img(self, imgNum, cmap = 'gray', figsize = (20, 20), printTitle = False):
        shape = (self.images.shape[1], self.images.shape[2])
        
        fig, axes = plt.subplots(figsize = figsize)
        if printTitle:
            axes.title.set_text(f'Reconstructed image of frame {imgNum}')

        axes.axis('off')
        
        axes.imshow(minmax_scale(self.pca.components_, axis = 1)[imgNum - 1].reshape(shape), cmap = cmap)

