import numpy as np
import os
from PIL import Image

def image_to_array(path_to_image , size = (64,64), if_gray = False):
    array_img = Image.open(path_to_image)
    if (if_gray):
        array_img = array_img.convert('L')
    else:
        array_img = array_img.convert('RGB')
    array_img=array_img.resize(size)
    return np.array(array_img).flatten()

def folder_of_images_to_array_of_images(folder_path,size=(64,64),if_gray=False):
    data=[]
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image_name)
        try:
            data.append(image_to_array(image_path,size,if_gray))
        except Exception as e:
            print(f'Błąd przy zdj {image_name}: {e}')
    return np.array(data)

