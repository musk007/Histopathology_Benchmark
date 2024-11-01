import sys
sys.path.insert(1, '/home/roba.majzoub/Histopathology_Benchmark')

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
# import staintools

# image_path = "/l/users/roba.majzoub/artifacts/535940-IMG012x028-2.JPG"
# # image_path = "/l/users/roba.majzoub/plip_data/TCGA-Uniform/Adrenocortical_carcinoma/5/TCGA-OR-A5J1-01Z-00-DX1/0_0_1013.jpg"
# dest = "/l/users/roba.majzoub/artifacts/outputs"

# image = Image.open(image_path)

# =============================================================================
# 1. Brightness and Contrast
# =============================================================================
def luminance (image_path, choice, req, im_name, dest):
    image = Image.open(image_path)
    os.makedirs(os.path.join(dest, "luminance"), exist_ok=True)
    # for c in range(1,10,2): #4 levels
    for c in range(7,10,7): #4 levels
        if req == "bright":
            c_cor = 1 + (c / 10)
        elif req == "dark":
            c_cor = 1 - (c / 10) 
        if choice == "brightness":
            image_enh = ImageEnhance.Brightness(image)
        if choice == "contrast":
            image_enh = ImageEnhance.Contrast(image)
        image_out = image_enh.enhance(c_cor)
        image_out.save(os.path.join(dest, f"luminance/{im_name}_{choice}_{req}_{c}.jpg"), quality=80)

# im_name = image_path.split("/")[-1].split(".")[0]
# luminance(image, "contrast", "bright", im_name)
# luminance(image, "contrast", "dark", im_name)
# luminance(image, "brightness", "bright", im_name)
# luminance(image, "brightness", "dark", im_name)


# =============================================================================
# Overlay Function
# =============================================================================
#Function to create overlay
def transparentOverlay(src, overlay, pos=(0,0), scale = 1, alpha_factor=1):
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
            alpha *= alpha_factor
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

# =============================================================================
# 2. Dark Spots
# =============================================================================
def spottify(image_path,im_name, size, dest):
    os.makedirs(os.path.join(dest, "spots"), exist_ok=True)
    if size =="small":
        spot_dir = "/l/users/roba.majzoub/artifacts/spots_small"
    elif size =="large": 
        spot_dir = "/l/users/roba.majzoub/artifacts/spots_big"
    #Load spots. Coordinates are for a random range of coordinates where spot can be applied to fully affect the  patch
    if size == "small":
        sp_1 = cv2.imread(os.path.join(spot_dir, 'SP1.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
        sp_2 = cv2.imread(os.path.join(spot_dir, 'SP2.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [0:150, 0:150]
        sp_3 = cv2.imread(os.path.join(spot_dir, 'SP3.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [-20:130, -20:130]
        sp_4 = cv2.imread(os.path.join(spot_dir, 'SP4.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [-100:0, -100:0]
    elif size == "large":
        sp_1 = cv2.imread(os.path.join(spot_dir, 'med_spot_2_2x.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
        sp_2 = cv2.imread(os.path.join(spot_dir, 'med_spot_2.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
        sp_3 = cv2.imread(os.path.join(spot_dir, 'med_spot.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
        sp_4 = cv2.imread(os.path.join(spot_dir, 'small_spot.png'), cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
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
        cv2.imwrite(os.path.join(dest, "spots/"+im_name +f"_spots_{size}_{i}"+".jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# spottify(image_path, im_name, "small")
# spottify(image_path, im_name, "large")

# =============================================================================
# 3. Elastic Deformations
# =============================================================================
def elastic(image_path, im_name, dest):
    os.makedirs(os.path.join(dest, "elastic"), exist_ok=True)
    image = cv2.imread(image_path)
    image = np.moveaxis(image, -1, 0)  
    for i in range (10,101,30):
            eltra = monai.transforms.Rand2DElastic(
                    spacing=(i,i),
                    magnitude_range=(1,1),
                    prob = 1.0)

            image_el = eltra(image, mode='nearest')
            image_el = np.moveaxis(image_el, 0, -1)
            
            # image_el = cv2.cvtColor(image_el, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(os.path.join(dest, "elastic/"+im_name +f"_elastic_{i}"+".jpg"), image_el, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
# elastic(image_path, im_name)

# =============================================================================
# 4. Fat Droplets
# =============================================================================
###############      Adding Fat      ################
fat_path = "/l/users/roba.majzoub/artifacts/fat.png"
def fatify(image_path, im_name, alpha_factor, dest):
    os.makedirs(os.path.join(dest, "fat"), exist_ok=True)
    image = cv2.imread(image_path)
    #Load oil drop image
    fat = cv2.imread(fat_path, cv2.IMREAD_UNCHANGED)
    image_out = transparentOverlay(image,fat,(0,0),1, alpha_factor)
    cv2.imwrite(os.path.join(dest, "fat/"+im_name +f"_fat_{alpha_factor}.jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# fatify(image_path, im_name, 0.5)
# fatify(image_path, im_name, 1)
# fatify(image_path, im_name, 1.5)
# fatify(image_path, im_name, 2)

# =============================================================================
# 6. Flipping
# =============================================================================

def flipper (image_path, im_name, severity, dest):
    image = cv2.imread(image_path)
    os.makedirs(os.path.join(dest, "flip"), exist_ok=True)
    if severity == 1:
        image_out = cv2.flip(image, 1)  # horizontal flip
    elif severity == 2:
        image_out = cv2.flip(image, 0)  # vertical flip
    elif severity == 3:
        image_out = cv2.flip(image, -1) # horizontal and vertical flip (180)
    elif severity == 4:
        flipped_temp = cv2.flip(image, 1)
        image_out = cv2.flip(flipped_temp, 1)
    cv2.imwrite(os.path.join(dest, "flip/"+im_name +f"_flipped_{severity}.jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
# flipper(image_path, im_name, 1)
# flipper(image_path, im_name, 2)
# flipper(image_path, im_name, 3)
# flipper(image_path, im_name, 4)

# =============================================================================
# 7. Gaussian Blur
# =============================================================================
#Number of gaussian levels to test
num_g_lev = 4
def blur(image_path, im_name, dest):
    image = cv2.imread(image_path)
    os.makedirs(os.path.join(dest, "blurr"), exist_ok=True)
    for i in range(1,2*num_g_lev,2):
        image_blur = cv2.GaussianBlur(image, (i, i), 0)
        cv2.imwrite(os.path.join(dest, "blurr/"+im_name +f"_{i}_blurred.jpg"), image_blur, [cv2.IMWRITE_JPEG_QUALITY, 80])


# blur(image_path, im_name)

# =============================================================================
# 8. Compression
# =============================================================================
def compress1(image_path, im_name, dest):
    image = cv2.imread(image_path)
    os.makedirs(os.path.join(dest, "compression"), exist_ok=True)
    for c in range (5,25,5):
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), c]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg,1)
        cv2.imwrite(os.path.join(dest, "compression/"+im_name +f"_{c}_compression_1.jpg"), decimg, [cv2.IMWRITE_JPEG_QUALITY, 80])


def compress2(image_path, im_name, quality, dest):
    # Load image
    image = Image.open(image_path)
    os.makedirs(os.path.join(dest, "compression"), exist_ok=True)

    # Save with high compression (low quality)
    image.save(os.path.join(dest,"compression/"+im_name+f"_{quality}_compression_2.jpg"), quality=10)

# compress1(image_path, im_name)
# compress2(image_path, im_name, 5)
# compress2(image_path, im_name, 10)
# compress2(image_path, im_name, 50)
# compress2(image_path, im_name, 70)

# =============================================================================
# 9. Rotation
# =============================================================================
###############      Rotate      ################

def rotator (image_path, im_name, severity, dest):
    image = cv2.imread(image_path)
    os.makedirs(os.path.join(dest, "rotate"), exist_ok=True)
    if severity == 1:
        image_out = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif severity == 2:
        image_out = cv2.rotate(image, cv2.ROTATE_180)
    elif severity == 3:
        image_out = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # else:
    #     image_out = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     image_out = cv2.rotate(image_out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(os.path.join(dest, "rotate/"+im_name +f"_rotated_{severity}.jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# rotator(image_path, im_name, 1)
# rotator(image_path, im_name, 2)
# rotator(image_path, im_name, 3)
# rotator(image_path, im_name, 4)

# =============================================================================
# 10. Squamous epithelia
# =============================================================================
def squamous(image_path, im_name, severity, dest):
    os.makedirs(os.path.join(dest, "squamous"), exist_ok=True)
    if severity in [1,2]:
        epit_dir = "/l/users/roba.majzoub/artifacts/epit_small"
    elif severity in [3,4]:
        epit_dir = "/l/users/roba.majzoub/artifacts/epit_big"
    #Load spots
    if severity in [1,2]:
        ep_1 = cv2.imread(os.path.join(epit_dir,'sq1.png'), cv2.IMREAD_UNCHANGED)
        ep_2 = cv2.imread(os.path.join(epit_dir,'sq2.png'), cv2.IMREAD_UNCHANGED)
        ep_3 = cv2.imread(os.path.join(epit_dir,'sq4.png'), cv2.IMREAD_UNCHANGED)
        ep_4 = cv2.imread(os.path.join(epit_dir,'sq5.png'), cv2.IMREAD_UNCHANGED)
        ep_5 = cv2.imread(os.path.join(epit_dir,'sq6.png'), cv2.IMREAD_UNCHANGED)
        ep_6 = cv2.imread(os.path.join(epit_dir,'sq7.png'), cv2.IMREAD_UNCHANGED)
        ep_7 = cv2.imread(os.path.join(epit_dir,'sq9.png'), cv2.IMREAD_UNCHANGED)
        ep_8 = cv2.imread(os.path.join(epit_dir,'sq10.png'), cv2.IMREAD_UNCHANGED)
    else:
        ep_1 = cv2.imread(os.path.join(epit_dir,'sq3.png'), cv2.IMREAD_UNCHANGED)
        ep_2 = cv2.imread(os.path.join(epit_dir,'sq8.png'), cv2.IMREAD_UNCHANGED)
        ep_3 = cv2.imread(os.path.join(epit_dir,'sq12.png'), cv2.IMREAD_UNCHANGED)
        ep_4 = cv2.imread(os.path.join(epit_dir,'sq16.png'), cv2.IMREAD_UNCHANGED)
        ep_5 = cv2.imread(os.path.join(epit_dir,'sq17.png'), cv2.IMREAD_UNCHANGED)
        ep_6 = cv2.imread(os.path.join(epit_dir,'sq18.png'), cv2.IMREAD_UNCHANGED)
        ep_7 = cv2.imread(os.path.join(epit_dir,'sq19.png'), cv2.IMREAD_UNCHANGED)
        ep_8 = cv2.imread(os.path.join(epit_dir,'sq20.png'), cv2.IMREAD_UNCHANGED)

    eps = [ep_1, ep_2, ep_3, ep_4, ep_5, ep_6, ep_7, ep_8]

    image = cv2.imread(image_path)
    #randomly select type of squamous epithelial complex
    ep_list = np.random.choice(len(eps), 1, replace=False)
    for i in ep_list:
        #randomly select coordinates at which it should appear (2x)
        he = randint (0,150)
        he = randint (0,150)
        wi = randint (0,150)
        wi = randint (0,150)
        image_out = transparentOverlay(image,eps[i],(he,wi),1)
        cv2.imwrite(os.path.join(dest, "squamous/"+im_name +f"_squamous_{severity}"+".jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])

# squamous(image_path, im_name, 1)
# squamous(image_path, im_name, 2)
# squamous(image_path, im_name, 3)
# squamous(image_path, im_name, 4)

# =============================================================================
# 11. Stain         NOT WORKING YET
# =============================================================================
stain_dir = '/l/users/roba.majzoub/artifacts/schemes_ready'
stain_types = sorted(os.listdir(stain_dir))
def stainer(image_path, im_name, dest):
    for stain_type in stain_types:
        st = staintools.read_image(stain_dir + stain_type)
        standardizer = staintools.BrightnessStandardizer()
        stain_norm = staintools.StainNormalizer(method='macenko')
        stain_norm.fit(st)
        
        path_result = path_result_gl + "_" + stain_type + ".txt"
        
        
        
        image = Image.open(image_path)
        im = np.array(image)
        #stain normalization
        im = standardizer.transform(im)
        
        try:
            im = stain_norm.transform(im)
            i=1
            print(f"stain transfer successful for {stain_type}")
        except:
            print("exception")
            i=0 #to control if stain transfer was possible for all patches
        print("ready")



# =============================================================================
# 11. Thread
# =============================================================================
def threader(image_path, im_name, size, dest):
    os.makedirs(os.path.join(dest, "thread"), exist_ok=True)
    if size == "small":
        thread_dir = "/l/users/roba.majzoub/artifacts/thread_small"
    if size == "large":
        thread_dir = "/l/users/roba.majzoub/artifacts/thread_big"
    #Load threads
    thread1 = cv2.imread(os.path.join(thread_dir,'thread_1.png'), cv2.IMREAD_UNCHANGED)
    thread2 = cv2.imread(os.path.join(thread_dir,'thread_2.png'), cv2.IMREAD_UNCHANGED)
    thread3 = cv2.imread(os.path.join(thread_dir,'thread_3.png'), cv2.IMREAD_UNCHANGED)
    thread4 = cv2.imread(os.path.join(thread_dir,'thread_4.png'), cv2.IMREAD_UNCHANGED)
    thread5 = cv2.imread(os.path.join(thread_dir,'thread_5.png'), cv2.IMREAD_UNCHANGED)
    thread6 = cv2.imread(os.path.join(thread_dir,'thread_6.png'), cv2.IMREAD_UNCHANGED)
    thread7 = cv2.imread(os.path.join(thread_dir,'thread_7.png'), cv2.IMREAD_UNCHANGED)
    thread8 = cv2.imread(os.path.join(thread_dir,'thread_8.png'), cv2.IMREAD_UNCHANGED)
    thread9 = cv2.imread(os.path.join(thread_dir,'thread_9.png'), cv2.IMREAD_UNCHANGED)
    thread10 = cv2.imread(os.path.join(thread_dir,'thread_10.png'), cv2.IMREAD_UNCHANGED)


    #Load masks
    mask1 = np.array(Image.open(os.path.join(thread_dir,'thread_1_mask.jpg')))
    mask2 = np.array(Image.open(os.path.join(thread_dir,'thread_2_mask.jpg')))
    mask3 = np.array(Image.open(os.path.join(thread_dir,'thread_3_mask.jpg')))
    mask4 = np.array(Image.open(os.path.join(thread_dir,'thread_4_mask.jpg')))
    mask5 = np.array(Image.open(os.path.join(thread_dir,'thread_5_mask.jpg')))
    mask6 = np.array(Image.open(os.path.join(thread_dir,'thread_6_mask.jpg')))
    mask7 = np.array(Image.open(os.path.join(thread_dir,'thread_7_mask.jpg')))
    mask8 = np.array(Image.open(os.path.join(thread_dir,'thread_8_mask.jpg')))
    mask9 = np.array(Image.open(os.path.join(thread_dir,'thread_9_mask.jpg')))
    mask10 = np.array(Image.open(os.path.join(thread_dir,'thread_10_mask.jpg')))
    

    masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10]
    threads = [thread1, thread2, thread3, thread4, thread5, thread6, thread7, thread8, thread9, thread10]

    image = cv2.imread(image_path)
    for i in range(len(threads)):
        temp = cv2.resize(threads[i], (image.shape[0],image.shape[1]), interpolation=cv2.INTER_AREA)
        threads[i] = temp
        temp = cv2.resize(masks[i], (image.shape[0],image.shape[1]), interpolation=cv2.INTER_AREA)
        masks[i] = temp

    i = randint (0, 9)
        
    image_over = transparentOverlay(image,threads[i],(0,0),1)
    
    image_blur = cv2.GaussianBlur(image_over, (5, 5), 0)

    # (0,0,0) is 3 dimensions of numpy and for every dimension 0 is threshold to be masked
    image_out = np.where(masks[i]==(0, 0, 0), image_over, image_blur)
    
    cv2.imwrite(os.path.join(dest, "thread/"+im_name +f"_{size}_threaded_{i}"+".jpg"), image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
# threader(image_path, im_name, "small")
# threader(image_path, im_name, "large")