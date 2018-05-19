import os, os.path
import cv2 as cv
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

class Back_ground:
    def __init__(self):
        self.local_name = {'grassland': [1,0,0,0,0,0,0], 'canyon':[0,1,0,0,0,0,0], 'forest':[0,0,1,0,0,0,0],
            'king\'s load':[0,0,0,1,0,0,0], 'lake':[0,0,0,0,1,0,0], 'desert':[0,0,0,0,0,1,0], 'ruin':[0,0,0,0,0,0,1]
        }
        self.answer = []

    def make_answer(self):
        for i in range (0, 3500):
            if i<500 :
                self.answer.append(self.local_name['grassland'])
            elif i<1000 :
                self.answer.append(self.local_name['canyon'])
            elif i<1500 :
                self.answer.append(self.local_name['forest'])
            elif i<2000 :
                self.answer.append(self.local_name['king\'s load'])
            elif i<2500 :
                self.answer.append(self.local_name['lake'])
            elif i<3000 :
                self.answer.append(self.local_name['desert'])
            elif i<3500 :
                self.answer.append(self.local_name['ruin'])
            else :
                print("not exist")


    def img_loader(self,category,number):
        DIR = 'img/'+category
        file_name = '/back_{}.png'.format(number)
        return cv.imread(DIR+file_name),file_name

    def nomalization(self,number):
        img,file_name = self.img_loader('background_re',number)
        img_resize = cv.resize(img, (623, 354),interpolation = cv.INTER_AREA)
        img_set = np.array([np.array(self.answer[number]),img_resize])
        ## load test    
        # if number%100 == 0:
        #     plt.imshow(img)
        #     plt.show()

        # img_resize.astype('float32')
        # img_resize = img_resize/255.0
        return img_set,file_name

    def save_image(self,image_set,file_name):
        DIR = 'img/background_set'
        cv.imwrite(DIR+file_name,image_set) #saved BGR




if __name__=='__main__':
    back_ground = Back_ground()
    back_ground.make_answer()
    for i in range(0,3500):
        img_set,file_name = back_ground.nomalization(i)
        back_ground.save_image(img_set,file_name)

    # np.savez('background.npz', image=X,answer=Y)
    ## check size of image
    # test_img = img_loader('background',600)
    # height, width, channels = test_img.shape
    # print (height, width, channels)
    # plt.figure()
    # plt.imshow(test_img)
    # test_img_resize = cv.resize(test_img, (623, 354),interpolation = cv.INTER_AREA)
    # plt.figure()
    # plt.imshow(test_img_resize)
    # plt.show()