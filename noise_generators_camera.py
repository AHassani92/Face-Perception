
# core utilities
import os
import shutil
import time
from glob import glob
import random
import numpy as np
import json
import multiprocessing as mp 
import math

# basic cv tools
import cv2 as cv

# feature extraction and classification
import sklearn
import skimage
from PIL import Image, ImageOps
from skimage import feature

def blur(image, sigma = None, IR = True):
        
    if sigma == None:
        sigma = random.randint(3, 5)
    image_noise = skimage.filters.gaussian(np.asarray(image), sigma=(sigma, sigma), truncate=3.5, channel_axis=2)
    image_noise = np.uint8(image_noise*255)


    return image_noise

def gaussian(image, var = None, IR = True, cv_image = False):
    
    if cv_image:
        image = Image.fromarray(image)
        
    if IR:
        image = image.convert('LA')
        
    if var == None:
        var = .01*random.randint(1, 3)
    image_noise = skimage.util.random_noise(np.asarray(image), mode='gaussian', seed=None, clip=True, var = var)
    image_noise = np.uint8(image_noise*255)
    
    if IR:        
        image_noise = Image.fromarray(image_noise)
        w, h = image_noise.size
        image_noise = image_noise.convert("RGB")
        image_noise = np.asarray(image_noise)


    return image_noise


def poisson(image, gauss = None, IR = True, cv_image = False):

    if cv_image:
        image = Image.fromarray(image)
        
    if IR:
        image = image.convert('LA')

    if gauss ==None:
        gauss = .001*random.randint(1, 3)
    image_noise = skimage.util.random_noise(np.asarray(image), mode='poisson', seed=None, clip=True)
    image_noise = np.uint8(image_noise*255)
    
    if IR:        
        image_noise = Image.fromarray(image_noise)
        w, h = image_noise.size
        image_noise = image_noise.convert("RGB")
        image_noise = np.asarray(image_noise)
        
    return image_noise


def salt_and_pepper(image, grain_amount = None, IR = True, cv_image = False):
    
    # if IR and PIL need to grayscale first or get weird issues
    if IR and not cv_image:
        image = ImageOps.grayscale(image)
        
    if grain_amount == None:
        grain_amount = .001 * random.randint(3, 6)
    image_noise = skimage.util.random_noise(np.asarray(image), mode='s&p', seed=None, clip=True, amount = grain_amount)
    image_noise = np.uint8(image_noise*255)
    return image_noise


def pepper(image, grain_amount = None, IR = True, cv_image = False):
    
    # if IR and PIL need to grayscale first or get weird issues
    if IR and not cv_image:
        image = ImageOps.grayscale(image)
        
    if grain_amount == None:
        grain_amount = .001 * random.randint(3, 6)
    image_noise = skimage.util.random_noise(np.asarray(image), mode='pepper', seed=None, clip=True, amount = grain_amount)
    image_noise = np.uint8(image_noise*255)
    return image_noise


def under_expose(image, gamma = None, environment_flag = False):
        
    if gamma == None:
        original_gamma = skimage.img_as_float(image).mean()
        
        if environment_flag:
            desired = original_gamma*.8
            floor = .1
        else:
            desired = .15
            floor = .03
            
        gamma = math.exp(1+original_gamma-desired)
    
    image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
    new_gamma = skimage.img_as_float(image_noise).mean()
    
    while (new_gamma > desired or new_gamma < floor):
            
        if new_gamma < desired:
            gamma -= .25*random.random()
        else:
            gamma += .25*random.random()
        
        image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
        new_gamma = skimage.img_as_float(image_noise).mean()
        #print(original_gamma, gamma, original_gamma-desired, new_gamma)
            
    return image_noise


def over_expose(image, gamma = None, environment_flag = False):
        
    if gamma == None:
        original_gamma = skimage.img_as_float(image).mean()

        if environment_flag:
            desired = original_gamma*1.2
            ceiling = .75
        else:
            desired = .85
            ceiling = .9
   
        gamma = math.exp(desired-original_gamma)-1
        if gamma <= 0: gamma = .1
    
    image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
    new_gamma = skimage.img_as_float(image_noise).mean()
    
    while (new_gamma < desired or new_gamma > ceiling):
            
        if new_gamma < desired:
            gamma -= .25*random.random()
        else:
            gamma += .25*random.random()
        
        if gamma <= 0: gamma = .1
        image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
        new_gamma = skimage.img_as_float(image_noise).mean()
        #print(original_gamma, gamma, original_gamma-desired, new_gamma)
    
    return image_noise

def image_write(noise_im, path, name, noise, counter = -1):
    noise_im = Image.fromarray(noise_im)
    if counter >= 0: count = '_'+str(counter)
    else: count = ''
    
    noise_im.save(path + name + '_' + noise.__name__ + count + '.png')