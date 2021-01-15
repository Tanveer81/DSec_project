import os
import steganogan
from steganogan import SteganoGAN
import random
import string

def get_random_alphanumeric_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str

steganogan = SteganoGAN.load(architecture='dense')

i = 0
for filename in sorted(os.listdir('../alaska2-image-steganalysis/Cover/')):
    if i%1000==0:
        print(i)
    i = i + 1
    steganogan.encode(f'../alaska2-image-steganalysis/Cover/{filename}',
                      f"../alaska2-image-steganalysis/SteganoGAN/{filename.replace('jpg', 'png')}",
                      get_random_alphanumeric_string(random.randint(10, 50)))