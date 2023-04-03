# Face-Perception
An open-source repository for our group's academic perception projects. This work is intended to for non-commercial research. If you use this repository, please cite the appropriate papers!

The following utilities are currently supported:
* Synthetic noise augmentation


## Noise Generators
To use the noise generators, clone the noise directory and modify the noise_faces.py script to match your data setup. Note that this uses the functions in the noise_generators_camera.py and noise_generators_environment.py files. Details are provided in the function comments.

### Sensor Noises
The following sensor noises are implimented:
  * poor_focus - sensor is blurry due to poor focus
  * dark_noise - photoreceptor leakage in the form of gaussian noise
  * shot_noise - randomized photon distribution as poisson function
  * salt_and_pepper - randomized analog to digital binarization error
  * under_expose - sensor does not expose sufficiently causing loss of features
  * over_expose - sensor exposes too much, saturating out features
  
### Environment Noises
The following environmental noises are implimented:
  * point_source - point source presents randomized saturated blob in image
  * point_shadow - small object presents randomized under exposed blob in image
  * streak_source - overhead source (e.g., sun) illuminates streak over top of image
  * streak_shadow - below horizon source (e.g., sun) illuminates streak over bottom of image
  * pipe_source - adjacent source (e.g., sun) illuminates pipe across middle of image
  * pipe_shadow - adjacent source (e.g., sun) is obstructed and casts shadow across middle of image
  
### Implimenting Noise Augmentations
The noise functions can be directly implimented by simply importing the functions into your data loader. Alternatively, a helper function is provided in the noise_faces.py file. This file provides scripts to both add noise (using multiprocessing for speed) or remove the images. Note, you will need to verify the parser matches your repository structure.
