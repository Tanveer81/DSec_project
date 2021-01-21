# DSec_project
TorchAttack: https://github.com/Harry24k/adversarial-attacks-pytorch

White Box Attack with ImageNet: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb

Notebooks:

**Evaluation**: https://colab.research.google.com/drive/1-SQPaTivdOtxq514WGOnofiUCxCcnWb3?usp=sharing (Add in git folder)


Documentation:

**Presentation**: https://docs.google.com/presentation/d/1nWF6kdSFXgJHggLFSHd_fG8sAkv-H3cqojQD-QzYeyw/edit#slide=id.p

**Final Paper**: https://www.overleaf.com/1311294973kbsxvtkcddxq


**Scripts**
1. steganogan_image_generation.py - This script creates steganoGAN stego from the cover images.
2. adversarial-attack.py This scripts performs adversarial attack for all 4 algorithms and also for cover and save the predictions of the analyzer on these adversarial and also the original images.
3. save_steganogan_adv_atk_images.py - This script saves the adversarial images for steganoGAN stegos for a small subset of the validation set (1000 images).
4. decode_secret.py - This script decodes and saves the secret from the adversarial images created from stegnoGAN adversarial images and also the original steganoGAN stegos.

To reprodice the results, one should run the four scripts in the same order. The data should be in the same directory in which scripts folder exists.



 
