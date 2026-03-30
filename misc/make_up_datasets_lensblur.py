import glob
import os
import sys
import random

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
from blurgenerator import lens_blur

SAVEPATH = './MFFdatasets-duts-tr'


def Resize(input):
    width, height = 640, 640
    img = input.resize((width, height))
    return img


def mask1(input):
    img = input.convert('RGB')
    for x in range(img.width):
        for y in range(img.height):
            data = img.getpixel((x, y))
            # if data[0] + data[1] + data[2] >= 128 and data[0] != 224 and data[1] != 224 and data[2] != 192:
            #     img.putpixel((x, y), (255, 255, 255))
            # else:
            #     img.putpixel((x, y), (0, 0, 0))
            if data[0] + data[1] + data[2] != 0:
                img.putpixel((x, y), (255, 255, 255))
    return img


def mask2(input):
    img = input.convert('RGB')
    for x in range(img.width):
        for y in range(img.height):
            data = img.getpixel((x, y))
            # if data[0] + data[1] + data[2] < 128:
            #     img.putpixel((x, y), (255, 255, 255))
            # elif data[0] == 224 and data[1] == 224 and data[2] == 192:
            #     img.putpixel((x, y), (255, 255, 255))
            # else:
            #     img.putpixel((x, y), (0, 0, 0))
            if data[0] + data[1] + data[2] == 0:
                img.putpixel((x, y), (255, 255, 255))
            else:
                img.putpixel((x, y), (0, 0, 0))
    return img


def LensBlur(input):
    blur_img = []

    blur_img.append(Image.fromarray(lens_blur(np.array(input), 5, 6, 3)))

    return blur_img


def Generate(input_path1, input_path2, filename, mode='train'):
    result_root1 = os.path.join(os.getcwd(), SAVEPATH, mode, 'sourceA\\')
    result_root2 = os.path.join(os.getcwd(), SAVEPATH, mode, 'sourceB\\')
    result_root3 = os.path.join(os.getcwd(), SAVEPATH, mode, 'decisionmap\\')
    result_root4 = os.path.join(os.getcwd(), SAVEPATH, mode, 'groundtruth\\')
    # Synthesized_imgA保存路径
    save_rootA_1 = result_root1 + filename[:-4]
    # Synthesized_imgB保存路径
    save_rootB_1 = result_root2 + filename[:-4]
    # 蒙版保存路径
    save_rootC_1 = result_root3 + filename[:-4]
    # 原图保存路径
    save_rootD_1 = result_root4 + filename[:-4]
    Original_img = Image.open(input_path1)
    Original_img = Original_img.convert('RGB')
    # Original_img = Resize(Original_img)
    Ground_img = Image.open(input_path2)
    # Ground_img = Resize(Ground_img)
    five_level_blur_list = LensBlur(Original_img)
    Mask1_img = mask1(Ground_img)
    Mask2_img = mask2(Ground_img)

    for Blurred_img in five_level_blur_list:
        Part_image_positive_1 = ImageChops.multiply(Original_img, Mask1_img)
        Part_image_positive_2 = ImageChops.multiply(Blurred_img, Mask2_img)
        Part_image_negative_1 = ImageChops.multiply(Blurred_img, Mask1_img)
        Part_image_negative_2 = ImageChops.multiply(Original_img, Mask2_img)
        Synthesized_imgA = ImageChops.add(Part_image_positive_1, Part_image_positive_2)
        Synthesized_imgB = ImageChops.add(Part_image_negative_1, Part_image_negative_2)
        Synthesized_imgA.save(save_rootA_1 + '_' + str(five_level_blur_list.index(Blurred_img)) + '.jpg')
        Synthesized_imgB.save(save_rootB_1 + '_' + str(five_level_blur_list.index(Blurred_img)) + '.jpg')
        Mask1_img.save(save_rootC_1 + '_' + str(five_level_blur_list.index(Blurred_img)) + '.png')
        Original_img.save(save_rootD_1 + '_' + str(five_level_blur_list.index(Blurred_img)) + '.jpg')


def DirCheck():
    if os.path.exists(SAVEPATH) is False:
        os.makedirs(SAVEPATH)

    if os.path.exists(os.path.join(SAVEPATH, 'train')) is False:
        os.makedirs(os.path.join(SAVEPATH, 'train'))
    if os.path.exists(os.path.join(SAVEPATH, 'train/sourceA')) is False:
        os.makedirs(os.path.join(SAVEPATH, 'train/sourceA'))
    if os.path.exists(os.path.join(SAVEPATH, 'train/sourceB')) is False:
        os.makedirs(os.path.join(SAVEPATH, 'train/sourceB'))
    if os.path.exists(os.path.join(SAVEPATH, 'train/decisionmap')) is False:
        os.makedirs(os.path.join(SAVEPATH, 'train/decisionmap'))
    if os.path.exists(os.path.join(SAVEPATH, 'train/groundtruth')) is False:
        os.makedirs(os.path.join(SAVEPATH, 'train/groundtruth'))

    # if os.path.exists(os.path.join(SAVEPATH, 'validate')) is False:
    #     os.makedirs(os.path.join(SAVEPATH, 'validate'))
    # if os.path.exists(os.path.join(SAVEPATH, 'validate/sourceA')) is False:
    #     os.makedirs(os.path.join(SAVEPATH, 'validate/sourceA'))
    # if os.path.exists(os.path.join(SAVEPATH, 'validate/sourceB')) is False:
    #     os.makedirs(os.path.join(SAVEPATH, 'validate/sourceB'))
    # if os.path.exists(os.path.join(SAVEPATH, 'validate/decisionmap')) is False:
    #     os.makedirs(os.path.join(SAVEPATH, 'validate/decisionmap'))
    # if os.path.exists(os.path.join(SAVEPATH, 'validate/groundtruth')) is False:
    #     os.makedirs(os.path.join(SAVEPATH, 'validate/groundtruth'))


def main(conifg):
    data_root = '../train_datasets'
    DirCheck()
    Ground_list_name = [i for i in os.listdir(os.path.join(data_root, 'DUTS-TR-Mask')) if i.endswith('png')]
    Ground_list = [os.path.join(data_root, 'DUTS-TR-Mask', i) for i in Ground_list_name]
    Original_list = [os.path.join(data_root, 'DUTS-TR-Image', i.split('.')[0] + '.jpg') for i in Ground_list_name]

    for i in tqdm(range(int(config.sp), int(config.ep)), file=sys.stdout):
        Generate(Original_list[i], Ground_list[i], Ground_list_name[i], 'train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp', type=str, default='0')
    parser.add_argument('--ep', type=str, default='9999')
    config = parser.parse_args()
    main(config)
