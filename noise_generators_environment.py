
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
from noise_generators_camera import blur, gaussian, poisson, salt_and_pepper, under_expose, over_expose, noise_image_write
# function to generate circular bright spots
def point_source(image, scale = None, randomize = True, IR = True):
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # randomize scale unless specified
    if randomize:
        scale_x = .01*random.randint(5,35)
        scale_y = .01*random.randint(5,35)
    elif scale == None:
        scale_x = .25
        scale_y = .25
    
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

    # apply masks and change the exposure
    blend = blur(over_expose(image, environment_flag = True), 2)
    img_over =  over_expose(image, environment_flag = True)
    img_under =  gaussian(under_expose(image, environment_flag = True), var = .001, cv_image = True, IR = IR)

    blend = cv.bitwise_and(blend, mask)
    spot = cv.bitwise_and(img_over, overlay)
    bg = cv.bitwise_and(img_under, background)

    image_point_source = bg + spot + blend
    
    return image_point_source

# function to generate circular shadows
def point_shadow(image, scale = None, randomize = True, IR = True):
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # randomize scale unless specified
    if randomize:
        scale_x = .01*random.randint(5,35)
        scale_y = .01*random.randint(5,35)
    elif scale == None:
        scale_x = .25
        scale_y = .25
    
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

    # apply masks and change the exposure
    blend = blur(under_expose(image, environment_flag = True), 2)
    img_over =  poisson(over_expose(image, environment_flag = True), cv_image = True, IR = IR)
    img_under =  gaussian(under_expose(image, environment_flag = True), var = .01, cv_image = True, IR = IR)

    blend = cv.bitwise_and(blend, mask)
    spot = cv.bitwise_and(img_under, overlay)
    bg = cv.bitwise_and(img_over, background)

    image_point_shadow = bg + spot + blend
    
    return image_point_shadow

# function to generate circular bright spots
def streak_source(image, randomize = True, IR = True):
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # randomize crop geometry
    top_left = [0, 0]
    top_right = [ww, 0]
    left_cut = random.randint(int(1*hh/4), int(3*hh/4))
    right_cut = random.randint(int(1*hh/4), int(3*hh/4))
    bottom_left = [0, left_cut]
    bottom_right = [ww, right_cut]
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
    
    # apply masks and change the exposure
    img_blur = blur(over_expose(image, environment_flag = True), 2)
    img_over =  over_expose(image, environment_flag = True)
    img_under =  gaussian(under_expose(image, environment_flag = True), var = .001, cv_image = True, IR = IR)

    blend = cv.bitwise_and(img_blur, blend)
    streak = cv.bitwise_and(img_over, overlay)
    bg = cv.bitwise_and(img_under, background)
    
    # assemble the new image
    image_streak_source = bg + streak + blend
    
    return image_streak_source

# function to generate circular bright spots
def streak_shadow(image, randomize = True, IR = True):
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # randomize crop geometry
    top_left = [0, 0]
    top_right = [ww, 0]
    left_cut = random.randint(int(1*hh/4), int(3*hh/4))
    right_cut = random.randint(int(1*hh/4), int(3*hh/4))
    bottom_left = [0, left_cut]
    bottom_right = [ww, right_cut]
    pts = np.array([top_left, top_right, bottom_right, bottom_left])
    
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
        
    # apply masks and change the exposure
    img_blur = blur(under_expose(image, environment_flag = True), 2)
    img_over =  poisson(over_expose(image, environment_flag = True), cv_image = True, IR = IR)
    img_under =  gaussian(under_expose(image, environment_flag = True), var = .01, cv_image = True, IR = IR)

    blend = cv.bitwise_and(img_blur, blend)
    streak = cv.bitwise_and(img_under, overlay)
    bg = cv.bitwise_and(img_over, background)
    
    # assemble the new image
    image_streak_shadow = bg + streak + blend
    
    return image_streak_shadow

# function to generate a light pipe across face
def pipe_source(image, randomize = True, IR = True):
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # randomize crop geometry
    top_left_cut = random.randint(int(.1*hh), int(hh/3))
    top_right_cut = random.randint(int(.1*hh), int(hh/3))
    bottom_left_cut = random.randint(int(hh/2*1.1), int(2/3*hh))
    bottom_right_cut = random.randint(int(hh/2*1.1), int(2/3*hh))
    
    top_left = [0, top_left_cut]
    top_right = [ww, top_right_cut]
    bottom_left = [0, bottom_left_cut]
    bottom_right = [ww, bottom_right_cut]
    
    pts = np.array([top_left, top_right, bottom_right, bottom_left])
    pts_top = np.array([[0,0], [ww, 0], top_right, top_left])
    pts_bottom = np.array([bottom_left, bottom_right, [ww, hh], [0,hh]])

    # bllend crop geometries
    blend_slice = .01*np.minimum(hh, ww)
    blend_top_left = [0, int(top_left_cut - blend_slice)]
    blend_top_right = [ww, int(top_right_cut - blend_slice)]
    blend_bottom_left = [0, int(bottom_left_cut + blend_slice)]
    blend_bottom_right = [ww, int(bottom_right_cut + blend_slice)]
    
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
    
    # apply masks and change the exposure
    img_blur = blur(over_expose(image, environment_flag = True), 2)
    img_over =  over_expose(image, environment_flag = True)
    img_under =  gaussian(under_expose(image, environment_flag = True), var = .001, cv_image = True, IR = IR)

    blend_top = cv.bitwise_and(img_blur, blend_top)
    blend_bottom = cv.bitwise_and(img_blur, blend_bottom)
    streak = cv.bitwise_and(img_over, overlay)
    bg = cv.bitwise_and(img_under, background)
    
    # assemble the new image
    image_pipe_light = bg + streak +  blend_top + blend_bottom
    
    return image_pipe_light


# function to generate a shadow pipe across face
def pipe_shadow(image, randomize = True, IR = True):
    
    # image dimensions
    hh, ww = image.shape[:2]
    
    # randomize crop geometry
    top_left_cut = random.randint(int(.1*hh), int(hh/3))
    top_right_cut = random.randint(int(.1*hh), int(hh/3))
    bottom_left_cut = random.randint(int(hh/2*1.1), int(2/3*hh))
    bottom_right_cut = random.randint(int(hh/2*1.1), int(2/3*hh))
    
    top_left = [0, top_left_cut]
    top_right = [ww, top_right_cut]
    bottom_left = [0, bottom_left_cut]
    bottom_right = [ww, bottom_right_cut]
    
    pts = np.array([top_left, top_right, bottom_right, bottom_left])
    pts_top = np.array([[0,0], [ww, 0], top_right, top_left])
    pts_bottom = np.array([bottom_left, bottom_right, [ww, hh], [0,hh]])

    # blend crop geometries
    blend_slice = .01*np.minimum(hh, ww)
    blend_top_left = [0, int(top_left_cut - blend_slice)]
    blend_top_right = [ww, int(top_right_cut - blend_slice)]
    blend_bottom_left = [0, int(bottom_left_cut + blend_slice)]
    blend_bottom_right = [ww, int(bottom_right_cut + blend_slice)]
    
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
    
    # apply masks and change the exposure
    img_blur = blur(under_expose(image, environment_flag = True), 2)
    img_over =  poisson(over_expose(image, environment_flag = True), cv_image = True, IR = IR)
    img_under =  gaussian(under_expose(image, environment_flag = True), var = .01, cv_image = True, IR = IR)


    blend_top = cv.bitwise_and(img_blur, blend_top)
    blend_bottom = cv.bitwise_and(img_blur, blend_bottom)
    streak = cv.bitwise_and(img_under, overlay)
    bg = cv.bitwise_and(img_over, background)
    
    # assemble the new image
    image_pipe_shadow = bg + streak +  blend_top + blend_bottom
    
    return image_pipe_shadow