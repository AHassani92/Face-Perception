
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
    """
    The blur function adds a randomized amount of blur to the image.
    By default, it uses a kernel intensity ranging between 3 and 5 but can be specified.
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param sigma: the blur intensity coefficient
    :type sigma: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
    
    # check if specified parameter, else default
    if sigma == None:
        sigma = random.randint(3, 5)
        
    # apply blur filter and convert back to uint8
    image_noise = skimage.filters.gaussian(np.asarray(image), sigma=(sigma, sigma), truncate=3.5, channel_axis=2)
    image_noise = np.uint8(image_noise*255)

    # return the noisy image
    return image_noise

def gaussian(image, var = None, IR = True, cv_image = False):
    """
    The gaussian function represents dark-noise or photo-receptor leakage. This is done with adding randomized gaussian noise.
    By default, it assumes a variance between .01 * 1 and .01 * 3 or can be specified.
    Note this function needs PIL format to work properly; flags are used to convert from numpy as necessary.
    
    :param image: the image to be noise-augmented
    :type image: PIL image or numpy array
    
    :param var: the noise variance coefficient
    :type var: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :param cv_image: flag to indicate whether numpy array or PIL image
    :type cv_image: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # if opencv image (numpy array) convert to PIL
    if cv_image:
        image = Image.fromarray(image)
        
    # if infrared image, convert format 
    if IR:
        image = image.convert('LA')
        
    # if variance not specified use randomized default
    if var == None:
        var = .01*random.randint(1, 3)
        
    # apply gaussian noise and convert back to uint8
    image_noise = skimage.util.random_noise(np.asarray(image), mode='gaussian', seed=None, clip=True, var = var)
    image_noise = np.uint8(image_noise*255)
    
    # if infrared, need to convert back for proper storage
    if IR:        
        image_noise = Image.fromarray(image_noise)
        w, h = image_noise.size
        image_noise = image_noise.convert("RGB")
        image_noise = np.asarray(image_noise)

    # return the noise-augmented image
    return image_noise


def poisson(image, gauss = None, IR = True, cv_image = False):
    """
    The poisson function represents shot-noise or irregular photon distributione. This is done with adding poisson noise.
    Note this function needs PIL format to work properly; flags are used to convert from numpy as necessary.
    
    :param image: the image to be noise-augmented
    :type image: PIL image or numpy array
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :param cv_image: flag to indicate whether numpy array or PIL image
    :type cv_image: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """

    # if opencv image (numpy array) convert to PIL
    if cv_image:
        image = Image.fromarray(image)
        
    # if infrared image, convert format 
    if IR:
        image = image.convert('LA')
    
    # add poisson noise and convert back to uint8
    image_noise = skimage.util.random_noise(np.asarray(image), mode='poisson', seed=None, clip=True)
    image_noise = np.uint8(image_noise*255)
        
    # if infrared, need to convert back for proper storage
    if IR:        
        image_noise = Image.fromarray(image_noise)
        w, h = image_noise.size
        image_noise = image_noise.convert("RGB")
        image_noise = np.asarray(image_noise)
        
    # return the noise-augmented image
    return image_noise


def salt_and_pepper(image, grain_amount = None, IR = True, cv_image = False):
    """
    The salt_and_pepper function represents analog to digital conversion error. This is done with randomly assigning 0 and 255 values.
    By default, it assumes a graininess level .001 * 3 and .001 * 6 or can be specified.
    Note this function needs PIL format to work properly; flags are used to convert from numpy as necessary.
    
    :param image: the image to be noise-augmented
    :type image: PIL image or numpy array
    
    :param grain_amount: the image graininess coefficient
    :type grain_amount: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :param cv_image: flag to indicate whether numpy array or PIL image
    :type cv_image: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # if IR and PIL need to grayscale first or get weird issues
    if IR and not cv_image:
        image = ImageOps.grayscale(image)
        
    # if graininess not specified use randomized default
    if grain_amount == None:
        grain_amount = .001 * random.randint(3, 6)
    
    # apply noise and convert back to uint8
    image_noise = skimage.util.random_noise(np.asarray(image), mode='s&p', seed=None, clip=True, amount = grain_amount)
    image_noise = np.uint8(image_noise*255)
        
    # return noise-augmented image
    return image_noise


def under_expose(image, gamma = None, environment_flag = False):
    """
    The under_expose function represents poor image contrast where features are lost due to lack of exposure. 
    This is done with iteratively adjusting gamma down until it meets the threshold, but is above a floor (necessary to ensure some contrast).
    By default, it assumes a gamma target of 0.15 or 0.8 * the input for environmental blending or can be specified.
    For defaults on environmental effects, see the noise_generators_environment.py file.
    
    :param image: the image to be noise-augmented
    :type image: PIL image or numpy array
    
    :param gamma: the target gamma
    :type gamma: integer, optional
    
    :param environment_flag: flag to indicate whether noise is being used for environmental effects blending or not
    :type environment_flag: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """

    # if gamma unspecified use the defaults
    if gamma == None:
        
        # calculate the baseline
        original_gamma = skimage.img_as_float(image).mean()
        
        # if environmental blending, do a simple relative decrease
        if environment_flag:
            desired = original_gamma*.8
            floor = .1
                
        # otherwise under expose to the point of lost features
        else:
            desired = .15
            floor = .03
            
        # calculate the deviation to be applied
        gamma = math.exp(1+original_gamma-desired)
    
    # adjust gamma in a do-while loop format
    # first apply gamma deviation and calculate the new gamma
    image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
    new_gamma = skimage.img_as_float(image_noise).mean()
    
    # if gamma does not meet targets, iteratively adjust it
    while (new_gamma > desired or new_gamma < floor):
        
        # if below floor, adjust correction slightly less
        if new_gamma < floor:
            gamma -= .25*random.random()
        
        # if above target, adjust correction slightly more
        elif new_gamma > desired:
            gamma += .25*random.random()
        
        # apply gamma deviation and re-calculate the new gamma
        image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
        new_gamma = skimage.img_as_float(image_noise).mean()
            
    # return noise-augmented image
    return image_noise


def over_expose(image, gamma = None, environment_flag = False):
    """
    The over_expose function represents poor image contrast where features are saturated due to too much exposure. 
    This is done with iteratively adjusting gamma up until it meets the threshold, but is below a ceiling (necessary to ensure some contrast).
    By default, it assumes a gamma target of 0.85 or 1.2 * the input for environmental blending or can be specified.
    For defaults on environmental effects, see the noise_generators_environment.py file.
    
    :param image: the image to be noise-augmented
    :type image: PIL image or numpy array
    
    :param gamma: the target gamma
    :type gamma: integer, optional
    
    :param environment_flag: flag to indicate whether noise is being used for environmental effects blending or not
    :type environment_flag: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """

    # if gamma unspecified use the defaults
    if gamma == None:
        
        # calculate the baseline
        original_gamma = skimage.img_as_float(image).mean()
        
        # if environmental blending, do a simple relative decrease
        if environment_flag:
            desired = original_gamma*1.2
            ceiling = .75
                
        # otherwise under expose to the point of lost features
        else:
            desired = .85
            floor = .9
            
        # calculate the deviation to be applied with enforcing non-negative corrections
        gamma = math.exp(desired-original_gamma)-1
        if gamma <= 0: gamma = .1
                
    # adjust gamma in a do-while loop format
    # first apply gamma deviation and calculate the new gamma
    image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
    new_gamma = skimage.img_as_float(image_noise).mean()
    
    # if gamma does not meet targets, iteratively adjust it
    while (new_gamma < desired or new_gamma > ceiling):
        
        # if below floor, adjust correction slightly more
        if new_gamma < desired:
            gamma -= .25*random.random()
        
        # if above ceiling, adjust correction slightly less
        elif new_gamma > ceiling:
            gamma += .25*random.random()
        
        # apply gamma deviation and re-calculate the new gamma
        image_noise = skimage.exposure.adjust_gamma(np.asarray(image), gamma)
        new_gamma = skimage.img_as_float(image_noise).mean()
            
    # return noise-augmented image
    return image_noise


def image_write(noise_im, im_dir, im_name, noise, counter = -1):
    """
    The image_write function automates writing the noise-augmented image to an appropriate directory. 
    
    :param noise_im: the noise-augmented image
    :type noise_im: numpy array
    
    :param im_dir: the path to the image write directory
    :type im_dir: string
    
    :param im_name: the name of the image to be writteny
    :type im_name: string
    
    :param noise: the noise augmentation function
    :type noise: function pointer
    
    :param counter: a counter to append to the image for automation
    :type counter: integer, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
    
    # convert image to PIL format
    noise_im = Image.fromarray(noise_im)
    
    # if a counter is specified, format it accordingly
    if counter >= 0: count = '_'+str(counter).zfill(4)
    else: count = ''
    
    # generate the path and write the image to disk
    write_path = im_dir + im_name + '_' + noise.__name__ + count + '.png'
    noise_im.save(write_path)
