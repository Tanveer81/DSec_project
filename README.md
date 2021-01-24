# DSec_project
TorchAttack: https://github.com/Harry24k/adversarial-attacks-pytorch

White Box Attack with ImageNet: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb

Notebooks:

**Evaluation**: https://colab.research.google.com/drive/1-SQPaTivdOtxq514WGOnofiUCxCcnWb3?usp=sharing (Add in git folder)


Documentation:

**Presentation**: https://docs.google.com/presentation/d/1nWF6kdSFXgJHggLFSHd_fG8sAkv-H3cqojQD-QzYeyw/edit#slide=id.p

**Final Paper**: https://www.overleaf.com/1311294973kbsxvtkcddxq

**Data**

Data should be downloaded from this link 'https://www.kaggle.com/c/alaska2-image-steganalysis/data' and stored in '/alaska2-image-steganalysis' folder. This folder contains following directories and files: 
      - Cover/ contains 75k unaltered images meant for use in training.
      - JMiPOD/ contains 75k examples of the JMiPOD algorithm applied to the cover images.
      - JUNIWARD/contains 75k examples of the JUNIWARD algorithm applied to the cover images.
      - UERD/ contains 75k examples of the UERD algorithm applied to the cover images.
      - Test/ contains 5k test set images. These are the images for which you are predicting.
      - sample_submission.csv contains an example submission in the correct format.

**Scripts**
1. steganogan_image_generation.py: This script creates steganoGAN stego from the cover images stored in this directory '/alaska2-image-steganalysis/Cover/'. It encodes randomly created alphanumeric strings of random length between 10 to 50 into all the cover images using encoder of the steganoGAN model and saves the encoded images here '.alaska2-image-steganalysis/SteganoGAN/'.

2. adversarial-attack.py: This scripts does the following tasks:
      - Loads only validation dataset of cover and all stegos.
      - Loads the model and weight '/best-checkpoint-143epoch.bin' for the independent steganalyzer.
      - Gets the prediction of the steganalyzer for these images.
      - Performs 8 different adversarial attack for all 4 stegos and cover images.
      - Gets the prediction of the stegos and cover after they go through all the attacks attacks.
      - Saves all the predictions as '/Evaluation/adv_attack/submission.csv' for further performance analysis.

3. save_steganogan_adv_atk_images.py: This script attacks the  steganoGAN stegos '/alaska2-image-steganalysis/SteganoGAN/' for a small subset of the validation set (1000 images) and saves the original and all images created by 8 adversarial attacks in this directory 'alaska2-image-steganalysis/SteganoGAN_Attk/'. 

4. decode_secret.py: It loads the images of original stegos and attacked stegos from '/alaska2-image-steganalysis/SteganoGAN_Attk/' and uses steganoGAN decoder to retrieve the secret and saves the secrets in this file '/Evaluation/secrets.csv'. 

To reproduce the results, one should run the four scripts in the same order.


**Environment**
 We need two python environments because steganogan works only on pytorch 1.0.0. But other libraries and models need newer version.
 1. Steganogan Environment: Only steganogan_image_generation.py needs to be run in this environemnt with python 3.6.
 The only package to install is steganogan by running the command: pip install steganogan. It will install all necessary libraries.
 
 2. Other Environment: All other scripts would run in this environemnt.
 The requirements are:
      - python 3.6
      - pytorch 1.5.0
      - torchattacks 2.12.1
      - scikit-learn 0.23.2
      - opencv 3.4.2
      - scikit-image 0.17.2
      - pandas 1.1.5
      - numpy 1.19.2
      - albumentations 0.5.2
      - matplotlib 3.3.3
      - torchvision 0.6.0
      - efficientnet-pytorch 0.7.0
