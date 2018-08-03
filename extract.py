# IMPORTS
import os
import pickle
import numpy as np
import math

# GLOBALS
data_directory = "/home/stormrage/Projects/kaggle_cascade/Data/"
#data_directory = "D:\\Projects\\kaggle_cascade\\Data\\"
image_width = 768
image_height = 768


def get_image_data():
    images_ref = []
    images_data = []
    images_mask = []
    minimized_index = 10
    current_index = 0
    list_os = os.listdir(data_directory + "train")
    list_os.sort()
    for filename in list_os:
        with open(data_directory + "train/" + filename, "rb") as imgText:
            tmp_img = imgText.read()
            b = bytearray(tmp_img, 'base64')
        images_ref.append(filename)
        images_mask.append([])
        images_data.append([filename, str(b)])
        current_index += 1
        if current_index == minimized_index and minimized_index != 0:
            break
    np.save(data_directory + "bundled_images", np.array(images_data, dtype=str))
    del images_data
    raw_data = open(data_directory + "train_ship_segmentations.csv", "r").readlines()
    current_index = 0
    for line_number in range(1, len(raw_data)):
        line = raw_data[line_number].split(",")
        point_bundle_temp = []
        masks = line[1].replace("\n", "").split()
        for x in range(0, len(masks), 2):
            point_bundle_temp += [t for t in range(int(masks[x]), int(masks[x]) + int(masks[x + 1]))]
        for p in range(len(point_bundle_temp)):
            point_bundle_temp[p] = [
                point_bundle_temp[p] - (int(math.floor(point_bundle_temp[p] / image_height)) * image_height),
                int(math.floor(point_bundle_temp[p] / image_height))]

        images_mask[images_ref.index(line[0])] += point_bundle_temp
        current_index += 1
        if current_index == minimized_index and minimized_index != 0:
            break
    images_mask = zip(images_ref, images_mask)
    with open(data_directory + 'hitmask', 'wb') as fp:
        pickle.dump(images_mask, fp)

    return images_mask


def extract_to_cnn(mask_size, step_size, reload=False):
    output_cnn = []
    if reload or not os.path.exists(data_directory + 'hitmask'):
        images = get_image_data()
    else:
        with open(data_directory + 'hitmask', 'rb') as fp:
            images = pickle.load(fp)
    if reload or not os.path.exists(data_directory + 'masks/mask_' + str(mask_size) + '_' + str(step_size)):
        for i in images:
            if len(i[1]) == 0:
                zero_out = True
            else:
                zero_out = False
            temp = []
            for y in range(0, image_height - mask_size, step_size):
                for x in range(0, image_width - mask_size, step_size):
                    if zero_out:
                        temp.append(0)
                    else:
                        for element in i[1]:
                            if x <= element[0] <=(x + mask_size):
                                if y <= element[1] <=(y + mask_size):
                                    temp.append(1)
                                    break
            output_cnn.append([i[0], temp])
        with open(data_directory + 'masks/mask_' + str(mask_size) + '_' + str(step_size), 'wb') as fp:
            pickle.dump(output_cnn, fp)
    else:
        with open(data_directory + 'masks/mask_' + str(mask_size) + '_' + str(step_size), 'rb') as fp:
            output_cnn = pickle.load(fp)
    return output_cnn
