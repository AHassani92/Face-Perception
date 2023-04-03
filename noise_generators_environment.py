
# core utilities
import os
import shutil
import time
from glob import glob
import random
import numpy as np
import json
import multiprocessing as mp 

# basic cv tools
import cv2 as cv

# feature extraction and classification
import sklearn
import skimage
from PIL import Image, ImageOps
from skimage import feature

# helper generators
from noise_generators_camera import poor_focus, dark_noise, shot_noise, salt_and_pepper, under_expose, over_expose

def point_source(image, scale = None, IR = True):
    """
    The point_source function models specular point sources.
    This is done with modelling randomized ellipses, which are over-exposed, under-exposing the background and blending the boundary.
    By default, it assumes a randomized ellipse size between .05 and .035 with random placement. These can alternatively be specified.
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param scale: the ellipse scale relative to the image size
    :type scale: float, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # image dimensions
    hh, ww = image.shape[:2]
    
    # if scale not specified use randomized defaults
    if scale == None:
        scale_x = .01*random.randint(5,35)
        scale_y = .01*random.randint(5,35)
    
    # define elliptical blobs
    radius_x = int(np.floor(scale_x*np.minimum(hh,ww)))
    radius_y = int(np.floor(scale_y*np.minimum(hh,ww)))
    axes = (radius_x, radius_y)
    angle = random.randint(0,360)
    
    # blending blob
    radius_blend_x = int(radius_x*1.05)
    radius_blend_y = int(radius_y*1.05)
    axes_blend = (radius_blend_x, radius_blend_y)
    
    # randomize the location
    radius = np.maximum(radius_x, radius_y)
    yc = random.randint(radius, hh - radius)
    xc = random.randint(radius, ww - radius)
    
    # draw filled ellipses in white on black background as masks
    mask = np.zeros_like(image)
    overlay = cv.ellipse(mask, (xc,yc), axes, angle, 0, 360, (255,255,255), -1, cv.LINE_AA)
    mask_blend = np.zeros_like(image)
    mask_blend = cv.ellipse(mask_blend, (xc,yc), axes_blend, angle, 0, 360, (255,255,255), -1, cv.LINE_AA)
    
    # subtract masks and make into single channel
    mask = cv.subtract(mask_blend, mask)
    background = cv.bitwise_not(mask_blend)

    # apply the exposure effects to the image
    blend = poor_focus(over_expose(image, environment_flag = True), 2)
    img_over =  over_expose(image, environment_flag = True)
    img_under =  dark_noise(under_expose(image, environment_flag = True), var = .001, cv_image = True, IR = IR)
    
    # mask the effects appropriately
    blend = cv.bitwise_and(blend, mask)
    spot = cv.bitwise_and(img_over, overlay)
    bg = cv.bitwise_and(img_under, background)
    
    # combine the masked effects back into a full image
    image_point_source = bg + spot + blend
    
    # return noise-augmented image
    return image_point_source

def point_shadow(image, scale = None, randomize = True, IR = True):
    """
    The point_shadow function models a single obstruction presenting an elliptical shadow.
    This is done with modelling randomized ellipses, which are under-exposed, over-exposing the background and blending the boundary.
    By default, it assumes a randomized ellipse size between .05 and .035 with random placement. These can alternatively be specified.
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param scale: the ellipse scale relative to the image size
    :type scale: float, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # if scale not specified use randomized defaults
    if scale == None:
        scale_x = .01*random.randint(5,35)
        scale_y = .01*random.randint(5,35)
    
    # define elliptical blobs
    radius_x = int(np.floor(scale_x*np.minimum(hh,ww)))
    radius_y = int(np.floor(scale_y*np.minimum(hh,ww)))
    axes = (radius_x, radius_y)
    angle = random.randint(0,360)
    
    # blending blob
    radius_blend_x = int(radius_x*1.05)
    radius_blend_y = int(radius_y*1.05)
    axes_blend = (radius_blend_x, radius_blend_y)
    
    # randomize the location
    radius = np.maximum(radius_x, radius_y)
    yc = random.randint(radius, hh - radius)
    xc = random.randint(radius, ww - radius)
    
    # draw filled ellipses in white on black background as masks
    mask = np.zeros_like(image)
    overlay = cv.ellipse(mask, (xc,yc), axes, angle, 0, 360, (255,255,255), -1, cv.LINE_AA)
    mask_blend = np.zeros_like(image)
    mask_blend = cv.ellipse(mask_blend, (xc,yc), axes_blend, angle, 0, 360, (255,255,255), -1, cv.LINE_AA)
    
    # subtract masks and make into single channel
    mask = cv.subtract(mask_blend, mask)
    background = cv.bitwise_not(mask_blend)

    # apply the exposure effects to the image
    blend = poor_focus(under_expose(image, environment_flag = True), 2)
    img_over =  shot_noise(over_expose(image, environment_flag = True), cv_image = True, IR = IR)
    img_under =  dark_noise(under_expose(image, environment_flag = True), var = .01, cv_image = True, IR = IR)
   
    # mask the effects appropriately
    blend = cv.bitwise_and(blend, mask)
    spot = cv.bitwise_and(img_under, overlay)
    bg = cv.bitwise_and(img_over, background)
    
    # combine the masked effects back into a full image
    image_point_shadow = bg + spot + blend
    
    # return noise-augmented image
    return image_point_shadow


def streak_source(image, streak_angle = None, IR = True):
    """
    The streak_source function models an overhead source that presents as a bright streak across the image.
    This is done with modelling randomized overhead sun angles to slice the image, over-exposing the top, under-exposing the bottom and blending the boundary.
    By default, it assumes a slice between 1/4 to 3/4 of the image height randomly selected on both sides. This can alterantively be specified with an angle (to come).
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param streak_angle: the angle at which to slice the image
    :type streak_angle: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # image dimensions
    hh, ww = image.shape[:2]
    
    # if no angle is specified, randomly generate the slice
    if streak_angle == None:    
        top_left = [0, 0]
        top_right = [ww, 0]
        left_cut = random.randint(int(1*hh/4), int(3*hh/4))
        right_cut = random.randint(int(1*hh/4), int(3*hh/4))
        bottom_left = [0, left_cut]
        bottom_right = [ww, right_cut]
        
    # otherwise calculate the specified slice
    else:
        
        # enforce streak to be overhead
        assert(streak_angle > 0 and streak_angle < 180)
        raise ValueError('Function to come')
    
    # generate the slice
    pts = np.array([top_left, top_right, bottom_right, bottom_left])
    
    # blend crop geometries
    blend_slice = .01*np.minimum(hh, ww)
    blend_left = [0, int(left_cut + blend_slice)]
    blend_right = [ww, int(right_cut + blend_slice)]
    pts_blend = np.array([top_left, top_right, blend_right, blend_left])
    
    # make mask
    mask = np.zeros_like(image)
    overlay = cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
    
    # make blending masks
    mask_blend = np.zeros_like(image)
    blend = cv.drawContours(mask_blend, [pts_blend], -1, (255, 255, 255), -1, cv.LINE_AA)

    # subtract masks and make into single channel
    blend = cv.bitwise_xor(mask_blend, mask)
    background = cv.bitwise_not(cv.bitwise_or(mask, mask_blend))
    
    # apply the exposure effects to the image
    img_blur = poor_focus(over_expose(image, environment_flag = True), 2)
    img_over =  over_expose(image, environment_flag = True)
    img_under =  dark_noise(under_expose(image, environment_flag = True), var = .001, cv_image = True, IR = IR)
    
    # mask the effects appropriately
    blend = cv.bitwise_and(img_blur, blend)
    streak = cv.bitwise_and(img_over, overlay)
    bg = cv.bitwise_and(img_under, background)
    
    # combine the masked effects back into a full image
    image_streak_source = bg + streak + blend
    
    # return noise-augmented image
    return image_streak_source

def streak_shadow(image, streak_angle = None, IR = True):
    """
    The streak_shadow function models a below horizon source that illuminates the bottom of the image, effectively shadowing the top.
    This is done with modelling randomized overhead sun angles to slice the image, under-exposing the top, over-exposing the bottom and blending the boundary.
    By default, it assumes a slice between 1/4 to 3/4 of the image height randomly selected on both sides. This can alterantively be specified with an angle (to come).
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param streak_angle: the angle at which to slice the image
    :type streak_angle: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # image dimensions
    hh, ww = image.shape[:2]
    
    # if no angle is specified, randomly generate the slice
    if streak_angle == None:    
        top_left = [0, 0]
        top_right = [ww, 0]
        left_cut = random.randint(int(1*hh/4), int(3*hh/4))
        right_cut = random.randint(int(1*hh/4), int(3*hh/4))
        bottom_left = [0, left_cut]
        bottom_right = [ww, right_cut]
        
    # otherwise calculate the specified slice
    else:
        
        # enforce streak to be overhead
        assert(streak_angle > 0 and streak_angle < 180)
        raise ValueError('Function to come')
    
    # blend crop geometries
    blend_slice = .01*np.minimum(hh, ww)
    blend_left = [0, int(left_cut - blend_slice)]
    blend_right = [ww, int(right_cut - blend_slice)]
    pts_blend = np.array([top_left, top_right, blend_right, blend_left])
    
    # make mask
    mask = np.zeros_like(image)
    overlay = cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
    
    # make blending mask
    mask_blend = np.zeros_like(image)
    blend = cv.drawContours(mask_blend, [pts_blend], -1, (255, 255, 255), -1, cv.LINE_AA)

    # subtract masks and make into single channel
    blend = cv.bitwise_xor(mask, mask_blend)
    background = cv.bitwise_not(cv.bitwise_or(mask, mask_blend))
        
    # apply the exposure effects to the image
    img_blur = poor_focus(under_expose(image, environment_flag = True), 2)
    img_over =  shot_noise(over_expose(image, environment_flag = True), cv_image = True, IR = IR)
    img_under =  dark_noise(under_expose(image, environment_flag = True), var = .01, cv_image = True, IR = IR)

    # mask the effects appropriately
    blend = cv.bitwise_and(img_blur, blend)
    streak = cv.bitwise_and(img_under, overlay)
    bg = cv.bitwise_and(img_over, background)
    
    # combine the masked effects back into a full image
    image_streak_shadow = bg + streak + blend
    
    # return the noise-augmented image
    return image_streak_shadow

def pipe_source(image, pipe_angle = None, IR = True):
    """
    The pipe_source function models an adjacent light source that presents as an illuminated pipe across the image.
    This is done with modelling randomized adjacent sun angles to create a pipe across the image, over-exposing the pipe, under-exposing the background and blending the boundaries.
    By default, it assumes a slice between 1/4 to 3/4 of the image height randomly selected on both sides. This can alterantively be specified with an angle (to come).
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param pipe_angle: the angle at which to create the image pipe
    :type pipe_angle: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # image dimensions
    hh, ww = image.shape[:2]
    
    # if no angle specified, randomly generate the pipe geometry
    if pipe_angle == None:
        top_left_cut = random.randint(int(.1*hh), int(hh/3))
        top_right_cut = random.randint(int(.1*hh), int(hh/3))
        bottom_left_cut = random.randint(int(hh/2*1.1), int(2/3*hh))
        bottom_right_cut = random.randint(int(hh/2*1.1), int(2/3*hh))

        top_left = [0, top_left_cut]
        top_right = [ww, top_right_cut]
        bottom_left = [0, bottom_left_cut]
        bottom_right = [ww, bottom_right_cut]
        
    # otherwise calculate the specified pipe
    else:
        
        # enforce streak to be overhead
        assert(pipe_angle > 0 and pipe_angle < 180)
        raise ValueError('Function to come')        
    
    # generate the pipe boundaries
    pts = np.array([top_left, top_right, bottom_right, bottom_left])
    pts_top = np.array([[0,0], [ww, 0], top_right, top_left])
    pts_bottom = np.array([bottom_left, bottom_right, [ww, hh], [0,hh]])

    # determine the crop blending points
    blend_slice = .01*np.minimum(hh, ww)
    blend_top_left = [0, int(top_left_cut - blend_slice)]
    blend_top_right = [ww, int(top_right_cut - blend_slice)]
    blend_bottom_left = [0, int(bottom_left_cut + blend_slice)]
    blend_bottom_right = [ww, int(bottom_right_cut + blend_slice)]
    
    # generate the blending boundaries
    pts_blend_top = np.array([[0,0], [ww,0], blend_top_right, blend_top_left])
    pts_blend_bottom = np.array([blend_bottom_left, blend_bottom_right, [ww, hh], [0,hh]])
        
    # make mask
    mask = np.zeros_like(image)
    overlay = cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
    
    # make blending masks
    mask_top = np.zeros_like(image)
    overlay_top = cv.drawContours(mask_top, [pts_top], -1, (255, 255, 255), -1, cv.LINE_AA)
    mask_bottom = np.zeros_like(image)
    overlay_bottom = cv.drawContours(mask_bottom, [pts_bottom], -1, (255, 255, 255), -1, cv.LINE_AA)
    mask_blend_top = np.zeros_like(image)
    blend_top = cv.drawContours(mask_blend_top, [pts_blend_top], -1, (255, 255, 255), -1, cv.LINE_AA)
    mask_blend_bottom = np.zeros_like(image)
    blend_bottom = cv.drawContours(mask_blend_bottom, [pts_blend_bottom], -1, (255, 255, 255), -1, cv.LINE_AA)
    
    # subtract masks and make into single channel
    blend_top = cv.bitwise_xor(mask_top, mask_blend_top)
    blend_bottom = cv.bitwise_xor(overlay_bottom, blend_bottom)
    total = cv.bitwise_or(cv.bitwise_or(blend_top, blend_bottom), mask)
    background = cv.bitwise_not(total)
    
    # apply the exposure effects to the image
    img_blur = poor_focus(over_expose(image, environment_flag = True), 2)
    img_over =  over_expose(image, environment_flag = True)
    img_under =  dark_noise(under_expose(image, environment_flag = True), var = .001, cv_image = True, IR = IR)

    # mask the effects appropriately
    blend_top = cv.bitwise_and(img_blur, blend_top)
    blend_bottom = cv.bitwise_and(img_blur, blend_bottom)
    streak = cv.bitwise_and(img_over, overlay)
    bg = cv.bitwise_and(img_under, background)
    
    # combine the masked effects back into a full image
    image_pipe_light = bg + streak +  blend_top + blend_bottom
    
    # return the noise-augmented image
    return image_pipe_light


def pipe_shadow(image, randomize = True, IR = True):
     """
    The pipe_shadow function models an adjacent obstruction that presents as an shadow pipe across the image.
    This is done with modelling randomized adjacent sun angles to create a pipe across the image, under-exposing the pipe, over-exposing the background and blending the boundaries.
    By default, it assumes a slice between 1/4 to 3/4 of the image height randomly selected on both sides. This can alterantively be specified with an angle (to come).
    
    :param image: the image to be noise-augmented
    :type image: numpy array
    
    :param pipe_angle: the angle at which to create the image pipe
    :type pipe_angle: integer, optional
    
    :param IR: flag to indicate whether infrared image or not
    :type IR: boolean, optional
    
    :return: the noise-augmented image
    :rtype: numpy array
    """
        
    # image dimensions
    hh, ww = image.shape[:2]
    

    # if no angle specified, randomly generate the pipe geometry
    if pipe_angle == None:
        top_left_cut = random.randint(int(.1*hh), int(hh/3))
        top_right_cut = random.randint(int(.1*hh), int(hh/3))
        bottom_left_cut = random.randint(int(hh/2*1.1), int(2/3*hh))
        bottom_right_cut = random.randint(int(hh/2*1.1), int(2/3*hh))

        top_left = [0, top_left_cut]
        top_right = [ww, top_right_cut]
        bottom_left = [0, bottom_left_cut]
        bottom_right = [ww, bottom_right_cut]
        
    # otherwise calculate the specified pipe
    else:
        
        # enforce streak to be overhead
        assert(pipe_angle > 0 and pipe_angle < 180)
        raise ValueError('Function to come')      
    
    # generate the pipe boundaries
    pts = np.array([top_left, top_right, bottom_right, bottom_left])
    pts_top = np.array([[0,0], [ww, 0], top_right, top_left])
    pts_bottom = np.array([bottom_left, bottom_right, [ww, hh], [0,hh]])

    # determine the crop blending points
    blend_slice = .01*np.minimum(hh, ww)
    blend_top_left = [0, int(top_left_cut - blend_slice)]
    blend_top_right = [ww, int(top_right_cut - blend_slice)]
    blend_bottom_left = [0, int(bottom_left_cut + blend_slice)]
    blend_bottom_right = [ww, int(bottom_right_cut + blend_slice)]
    
    # generate the blending boundaries
    pts_blend_top = np.array([[0,0], [ww,0], blend_top_right, blend_top_left])
    pts_blend_bottom = np.array([blend_bottom_left, blend_bottom_right, [ww, hh], [0,hh]])
        
    # make mask
    mask = np.zeros_like(image)
    overlay = cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
    
    # make blending masks
    mask_top = np.zeros_like(image)
    overlay_top = cv.drawContours(mask_top, [pts_top], -1, (255, 255, 255), -1, cv.LINE_AA)
    mask_bottom = np.zeros_like(image)
    overlay_bottom = cv.drawContours(mask_bottom, [pts_bottom], -1, (255, 255, 255), -1, cv.LINE_AA)
    mask_blend_top = np.zeros_like(image)
    blend_top = cv.drawContours(mask_blend_top, [pts_blend_top], -1, (255, 255, 255), -1, cv.LINE_AA)
    mask_blend_bottom = np.zeros_like(image)
    blend_bottom = cv.drawContours(mask_blend_bottom, [pts_blend_bottom], -1, (255, 255, 255), -1, cv.LINE_AA)
    
    # subtract masks and make into single channel
    blend_top = cv.bitwise_xor(mask_top, mask_blend_top)
    blend_bottom = cv.bitwise_xor(overlay_bottom, blend_bottom)
    total = cv.bitwise_or(cv.bitwise_or(blend_top, blend_bottom), mask)
    background = cv.bitwise_not(total)
    
    # apply the exposure effects to the image
    img_blur = poor_focus(under_expose(image, environment_flag = True), 2)
    img_over =  shot_noise(over_expose(image, environment_flag = True), cv_image = True, IR = IR)
    img_under =  dark_noise(under_expose(image, environment_flag = True), var = .01, cv_image = True, IR = IR)

    # mask the effects appropriately
    blend_top = cv.bitwise_and(img_blur, blend_top)
    blend_bottom = cv.bitwise_and(img_blur, blend_bottom)
    streak = cv.bitwise_and(img_under, overlay)
    bg = cv.bitwise_and(img_over, background)
    
    # combine the masked effects back into a full image
    image_pipe_shadow = bg + streak +  blend_top + blend_bottom
    
    # return the noise-augmented image
    return image_pipe_shadow
