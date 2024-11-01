#######  Code adopted from https://github.com/cpath-ukk/Artifact/tree/main
####### Original author Yuri Tolkach / some changes were made to fit the project


import os
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import random
from random import randint
import cv2
import monai

image_path = "/l/users/roba.majzoub/plip_data/TCGA-Uniform/Adrenocortical_carcinoma/5/TCGA-OR-A5J1-01Z-00-DX1/0_0_1013.jpg"
dest = "/l/users/roba.majzoub/artifacts/outputs"

image = Image.open(image_path)

# =============================================================================
# 1. Brightness and Contrast
# =============================================================================
def luminance (image, choice, req, im_name):
    for c in range(1,10,3): #4 levels
        if req == "bright":
            c_cor = 1 + (c / 10)
        elif req == "dark":
            c_cor = 1 - (c / 10) 
        if choice == "brightness":
            image_enh = ImageEnhance.Brightness(image)
        if choice == "contrast":
            image_enh = ImageEnhance.Contrast(image)
        image_out = image_enh.enhance(c_cor)
        image_out.save(os.path.join(dest, f"{im_name}_{choice}_{req}_{c}.jpg"), quality=80)

im_name = image_path.split("/")[-1].split(".")[0]
# luminance(image, "contrast", "bright", im_name)
# luminance(image, "contrast", "dark", im_name)
# luminance(image, "brightness", "bright", im_name)
# luminance(image, "brightness", "dark", im_name)


# =============================================================================
# 2. Dark Spots
# =============================================================================
spot_dir = "/l/users/roba.majzoub/artifacts/spots_small"
#Function to create overlay
def transparentOverlay(src, overlay, pos=(0,0), scale = 1):
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

# =============================================================================
# 3. Spots
# =============================================================================
def spottify(image_path,im_name):
    #Load spots. Coordinates are for a random range of coordinates where spot can be applied to fully affect the  patch
    sp_1 = cv2.imread(os.path.join(spot_dir, 'SP1.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
    sp_2 = cv2.imread(os.path.join(spot_dir, 'SP2.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [0:150, 0:150]
    sp_3 = cv2.imread(os.path.join(spot_dir, 'SP3.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [-20:130, -20:130]
    sp_4 = cv2.imread(os.path.join(spot_dir, 'SP4.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [-100:0, -100:0]
    sp_lst = [sp_1, sp_2, sp_3, sp_4]

    x_coord = [30, 0, -20, -100]
    y_coord = [180, 150, 130, 90, 0]

    for i in range (0,4,1):
        image = cv2.imread(image_path)
        he = randint (x_coord[i], y_coord[i])
        he = randint (x_coord[i], y_coord[i])
        wi = randint (x_coord[i], y_coord[i])
        wi = randint (x_coord[i], y_coord[i])
        image_out = transparentOverlay(image,sp_lst[i],(he,wi),1)
        image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dest, im_name +f"_spots_{i}"+".jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# spottify(image_path)

# =============================================================================
# 4. Elastic Deformations
# =============================================================================
def elastic(image_path, im_name):
    image = cv2.imread(image_path)
    image = np.moveaxis(image, -1, 0)  
    for i in range (10,101,20):
            eltra = monai.transforms.Rand2DElastic(
                    spacing=(i,i),
                    magnitude_range=(1,1),
                    prob = 1.0)

            image_el = eltra(image, mode='nearest')
            image_el = np.moveaxis(image_el, 0, -1)
            
            # image_el = cv2.cvtColor(image_el, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(os.path.join(dest, im_name +f"_elastic_{i}"+".jpg"), image_el, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
# elastic(image_path, im_name)

# =============================================================================
# 5. Fat Droplets
# =============================================================================
###############      Adding Fat      ################
fat_path = "/l/users/roba.majzoub/artifacts/fat.png"
def fatify(image_path, im_name):
    image = cv2.imread(image_path)
    #Load oil drop image
    fat = cv2.imread(fat_path, cv2.IMREAD_UNCHANGED)
    image_out = transparentOverlay(image,fat,(0,0),1)
    cv2.imwrite(os.path.join(dest, im_name +"_fat.jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# fatify(image_path, im_name)

# =============================================================================
# 6. Flipping
# =============================================================================

def flipper (image_path, im_name):
    image = cv2.imread(image_path)
    rand_num = random.randint(1,3)
    if rand_num == 1:
        image_out = cv2.flip(image, 0)
    if rand_num == 2:
        image_out = cv2.flip(image, 1)
    else:
        image_out = cv2.flip(image, -1)
    cv2.imwrite(os.path.join(dest, im_name +"_flipped.jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
# flipper(image_path, im_name)

# =============================================================================
# 7. Gaussian Blur
# =============================================================================
#Number of gaussian levels to test
num_g_lev = 4
def blur(image_path, im_name):
    image = cv2.imread(image_path)
    for i in range(1,2*num_g_lev,2):
        image_blur = cv2.GaussianBlur(image, (i, i), 0)
        cv2.imwrite(os.path.join(dest, im_name +f"_{i}_blurred.jpg"), image_blur, [cv2.IMWRITE_JPEG_QUALITY, 80])


# blur(image_path, im_name)

# =============================================================================
# 8. Compression
# =============================================================================
def compress1(image_path, im_name):
    image = cv2.imread(image_path)
    for c in range (5,10,5):
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), c]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg,1)
        cv2.imwrite(os.path.join(dest, im_name +f"_{c}_compression_1.jpg"), decimg, [cv2.IMWRITE_JPEG_QUALITY, 80])


def compress2(image_path, im_name):
    # Load image
    image = Image.open(image_path)

    # Save with high compression (low quality)
    image.save(os.path.join(dest,im_name+"_compression_2.jpg"), quality=10)

# compress1(image_path, im_name)
# compress2(image_path, im_name)

# =============================================================================
# 9. Rotation
# =============================================================================
###############      Rotate      ################

def rotator (image_path, im_name):
    image = cv2.imread(image_path)
    rand_num = random.randint(1,3)
    if rand_num == 1:
        image_out = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rand_num == 2:
        image_out = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        image_out = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(dest, im_name +"_rotated.jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# rotator(image_path, im_name)

# =============================================================================
# 10. Squamous epithelia
# =============================================================================
