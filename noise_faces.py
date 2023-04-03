
# core utilities
import os
import shutil
import time
from glob import glob
import random
import numpy as np
import json
import multiprocessing as mp 
import argparse

# basic cv tools
import cv2 as cv
from PIL import Image, ImageOps

# noise generators
from noise_generators_camera import poor_focus, dark_noise, shot_noise, salt_and_pepper, under_expose, over_expose
from noise_generators_environment import point_source, point_shadow, streak_source, streak_shadow, pipe_source, pipe_shadow


# argument parsing globals
MODE = ['NOISE', 'RM']
NOISE = ['ALL', 'CAM', 'ENV']


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


def parse_args():
    """
    The parse_args function parses the program inputs. This determines whether to noise or remove images as well as what noise type.
    
    :return: the parsed program inputs
    :rtype: dictionary
    """
    
    # create the argument parser
    parser = argparse.ArgumentParser(
        description='Liveliness BJL Argument Parser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    # add argument for mode
    parser.add_argument('-m', '--mode', type=str, default='NOISE',
                        help='Noisify or remove noisy data.',
                        choices=MODE
                       )
    
    # add argument for noise type
    parser.add_argument('-n', '--noise', type=str, default='ALL',
                        help='Synthetic noise type.',
                        choices=NOISE
                       )
    
    # parse the program input arguments
    args = parser.parse_args()

    # return the parsed dictionary
    return args

def noise_helper(im_dir, write_loc, camera_flag = True, environment_flag = True):
    """
    The noise_helper function breaks out the directory noise-augmentation logic for multiprocessin.
    
    :param im_dir: the directory of the where the data is
    :type im_dir: string
    
    :param write_loc: the path where the images should be written
    :type write_loc: string
    
    :param camera_flag: flag whether camera noises are to be used
    :type camera_flag: boolean
    
    :param environment_flag: flag whether environment noises are to be used
    :type environment_flag: boolean
    """
    
    features = {}

    # camera noises
    sensor_noises = [blur, gaussian, poisson, salt_and_pepper]
    exposures = [under_expose, over_expose]

    # environmental noises
    environments = [point_source, point_shadow, streak_source, streak_shadow, pipe_source, pipe_shadow]

    # total noise sources
    noises = []
    if camera_flag: noises += sensor_noises + exposures
    if environment_flag: noises += environments

    # setup the directory read and write paths
    loc_dir = os.path.join(person_data, im_dir)
    write_dir = os.path.join(write_loc, im_dir)

    # verify we have data first
    if os.path.isdir(loc_dir):

        # make the participant write directory if does not exist
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        # make the noise write subdirectory if does not exist
        for noisify in noises:
            if not os.path.exists(os.path.join(write_dir, noisify.__name__)):
                os.makedirs(os.path.join(write_dir, noisify.__name__))        

        # set the local directory
        os.chdir(loc_dir)
        
        # get the images
        images = glob("*.png")
        print('Noisifying', loc_dir, len(images))
        
        # iterate through images and noisify them
        for im in images:       
            
            # read in the image
            im_path = os.path.join(loc_dir, im)
            image = Image.open(im_path, 'r')
            
            # apply the noise-augmentation function
            noisify = random.choice(['none']+noises)

            # encofrce there is at least one valie noise
            if noisify != 'none':

                # environmental noises are CV operations
                if noisify in environments:
                    image = np.asarray(image)
                
                # apply the noise augmentation
                im_noisy = noisify(image)
                
                # write the image
                im_name = im.split(".")
                noise_image_write(im_noisy, gen_os_path(write_dir, noisify.__name__), im_name[0], noisify)


def noisify_data(data_root, write_root = '', camera_flag = True, environment_flag = True):
    """
    The noisify_data function parses through the data and calls the helper to do multi-processing augmentation.
    NOTE: MUST EDIT THIS TO MATCH YOUR DATASET STRUCTURE.
        
    :param data_root: the path to the dataset root
    :type data_root: string
    
    :param write_root: the path where the noise-augmented dataset should be written
    :type write_root: string
    
    :param camera_flag: flag whether camera noises are to be used
    :type camera_flag: boolean
    
    :param environment_flag: flag whether environment noises are to be used
    :type environment_flag: boolean
    
    :raises ValueError: asserts data path exists
    """
        
    # verify the directory exists
    assert(os.path.exists(data))
    
    # set the local database root
    proj_root = os.getcwd()

    # set the local directory
    os.chdir(data_root)

    # iterate through the directories
    directories = glob("*/")
    directories = sorted(directories)

    # default the write root to the data root
    if write_root == '': 
        write_root = data_root

    # otherwise verify the directory exists
    else:
        if not os.path.exists(write_root):
            os.makedirs(write_root)

    # setup the multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    # iterate through the directories
    # NOTE: YOU MAY NEED TO ADJUST THIS TO YOUR DATASET
    for num, person in enumerate(directories):
        
        # generate the local paths
        person_data = os.path.join(data_root, person)
        write_loc = os.path.join(write_root, person)
        pool.apply_async(noise_helper, args=(person_data, write_loc, camera_flag, environment_flag))
    
    # close the pool
    pool.close()
    pool.join()
            
    # reset the root directory
    os.chdir(proj_root)

def reset_noises(write_root):
    """
    The reset_noises function deletes the noisy data. This is useful if using generic data loaders that do not have knowledge of clean vs noisy data.
    NOTE: MUST EDIT THIS TO MATCH YOUR DATASET STRUCTURE.
    
    :param write_root: the path where the noise-augmented dataset is stored
    :type write_root: string
    
    :raises ValueError: asserts write_root path exists
    """
        
    # set the local database root
    proj_root = os.getcwd()

    # set the local directory
    os.chdir(write_root)

    # iterate through the directories
    directories = glob("*/")
    directories = sorted(directories)

    # noise directories
    camera_noises = [blur, gaussian, poisson, salt_and_pepper]
    exposures = [under_expose, over_expose]
    noises = camera_noises + exposures

    # iterate through the participants
    for num, person in enumerate(directories):
        
        # generate the write directory path
        write_dir = os.path.join(write_root,person)
        print('Removing data:', write_dir)

        # delete the noise subdirectories
        for noisify in noises:
            
            # generate the noise-subdirectory
            delete_dir = gen_os_path(write_dir, noisify.__name__)
            
            # delete if exists
            if os.path.exists(delete_dir):
                shutil.rmtree(delete_dir)

        
    # reset the root directory
    os.chdir(proj_root)

                        
if __name__ == "__main__":

    # absolute paths to where the data is stored
    # NOTE: MUST UPDATE THIS TO MATCH YOUR DATASET
    data = '/path/to/stored_data/'
    write_data = '/path/to/stored_data/'

    # get the arguments
    args = parse_args()

    # synthetic noise argument parsing
    if args.noise == 'ALL':
            camera_flag = True
            environment_flag = True
    elif args.noise == 'CAM':
            camera_flag = True
            environment_flag = False
    elif args.noise == 'ENV':
            camera_flag = False
            environment_flag = True
    
    # if mode is noise-augmentation, call the noisify function
    if args.mode == 'NOISE':        
        noisify_data(data, write_data, camera_flag, environment_flag)
        
    # if mode is removal, call the reset function
    elif args.mode == 'RM':
        reset_noises(write_data)
