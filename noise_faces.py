
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
from noise_generators_camera import blur, gaussian, poisson, salt_and_pepper, under_expose, over_expose, noise_image_write
from noise_generators_environment import point_source, point_shadow, streak_source, streak_shadow, pipe_source, pipe_shadow


MODE = ['NOISE', 'RM']
NOISE = ['ALL', 'CAM', 'ENV']

# program input parser
def parse_args():
    parser = argparse.ArgumentParser(
        description='Liveliness BJL Argument Parser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-m', '--mode', type=str, default='NOISE',
                        help='Noisify or remove noisy data.',
                        choices=MODE)
    parser.add_argument('-n', '--noise', type=str, default='ALL',
                        help='Synthetic noise type.',
                        choices=NOISE)
    args = parser.parse_args()

    return args

# utility to correct path to the OS
def gen_os_path(data, sub_dir):
    if os.name == 'nt':
        return data + sub_dir + '\\'
    else:
        return data + sub_dir + '/'

# helper function for multiprocessing
def noise_helper(person_data, write_loc, camera_flag = True, environment_flag = True):
    
    features = {}
    im_dirs = ['Live',  'Paper_Mask', 'Covid_Mask', 'Display_Replay', 'Spandex_Mask']
    im_dirs += ['Live_exterior',  'Paper_Mask_exterior', 'Covid_Mask_exterior', 'Display_Replay_exterior', 'Spandex_Mask_exterior']

    # camera noises
    sensor_noises = [blur, gaussian, poisson, salt_and_pepper]
    exposures = [under_expose, over_expose]

    # environmental noises
    environments = [point_source, point_shadow, streak_source, streak_shadow, pipe_source, pipe_shadow]

    # total noise sources
    noises = []
    if camera_flag: noises += sensor_noises + exposures
    if environment_flag: noises += environments

    # original annotated data
    for im_dir in im_dirs:
        
        loc_dir = gen_os_path(person_data, im_dir)
        write_dir = gen_os_path(write_loc, im_dir)

        # verify we have data first
        if os.path.isdir(loc_dir):

            # make the participant write directory if does not exist
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)

            # make the noise write subdirectory if does not exist
            for noisify in noises:
                if not os.path.exists(gen_os_path(write_dir, noisify.__name__)):
                    os.makedirs(gen_os_path(write_dir, noisify.__name__))        

            # set the local directory
            os.chdir(loc_dir)

            images = glob("*.png")
            print('Noisifying', loc_dir, len(images))
            # iterate through images and noisify them
            for im in images:       
                im_path = loc_dir+ im
                image = Image.open(im_path, 'r')
                
                noisify = random.choice(['none']+noises)

                # if we are going to noisy the data
                if noisify != 'none':

                    # environmental noises are CV operations
                    if noisify in environments:
                        image = np.asarray(image)

                    im_noisy = noisify(image)
                    im_name = im.split(".")
                    noise_image_write(im_noisy, gen_os_path(write_dir, noisify.__name__), im_name[0], noisify)
                '''

                for noisify in noises:
                    im_noisy = noisify(image)
                    noise_image_write(im_noisy, gen_os_path(write_dir, noisify.__name__), im, noisify)
                '''

# extract the features and write them to a json file for re-use
def noisify_data(data_root, write_root = '', camera_flag = True, environment_flag = True):
        
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

    pool = mp.Pool(mp.cpu_count())
    #random.shuffle(directories)
    for num, person in enumerate(directories):

        person_data = data_root+person
        write_loc = write_root+person
        #feature_helper(person_data, feature_fn)
        pool.apply_async(noise_helper, args=(person_data, write_loc, camera_flag, environment_flag))
        #noise_helper(person_data, write_loc, camera_flag, environment_flag)

    pool.close()
    pool.join()
            
    # reset the root directory
    os.chdir(proj_root)

# delete the noisy data for fresh generation
def reset_noises(write_root):
        
    # set the local database root
    proj_root = os.getcwd()

    # set the local directory
    os.chdir(write_root)

    # iterate through the directories
    directories = glob("*/")
    directories = sorted(directories)

    # liveliness directories
    im_dirs = ['Live',  'Paper_Mask', 'Covid_Mask', 'Display_Replay', 'Spandex_Mask']

    # noise directories
    camera_noises = [blur, gaussian, poisson, salt_and_pepper]
    exposures = [under_expose, over_expose]
    noises = camera_noises + exposures

    # iterate through the participants
    for num, person in enumerate(directories):

        write_loc = write_root+person
        print('Removing data:', write_loc)

        # go through the liveliness directories
        for im_dir in im_dirs:
            
            write_dir = gen_os_path(write_loc, im_dir)

            # delete the noise subdirectories
            for noisify in noises:
                delete_dir = gen_os_path(write_dir, noisify.__name__)
                if os.path.exists(delete_dir):
                    shutil.rmtree(delete_dir)

        
    # reset the root directory
    os.chdir(proj_root)

                        
if __name__ == "__main__":

    # determine the data path
    if os.name == 'nt':
        data = 'G:\\Anti-Spoof-Crops\\'
        write_data = 'G:\\Anti-Spoof-Crops-Noisy\\'
        print('Windows paths')
    else:
        data = '/media/ali/New Volume/Anti-Spoof-Crops/'
        print('Linux paths')

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

    if args.mode == 'NOISE':
        noisify_data(data, write_data, camera_flag, environment_flag)
    elif args.mode == 'RM':
        reset_noises(write_data)