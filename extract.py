# IMPORTS
import os
import pickle

# GLOBALS
data_directory = "D:\\Projects\\kaggle_cascade\\Data\\"
image_width = 768
image_height = 768


def get_image_data():

    images = []
    minimized_index = 0
    current_index = 0
    for filename in os.listdir(data_directory + "train\\"):
        images.append([filename, []])
        current_index += 1
        if current_index == minimized_index and minimized_index != 0:
            break
    raw_data = open(data_directory + "train_ship_segmentations.csv", "r").readlines()
    current_index = 0
    for line_number in range(1, len(raw_data)):
        line = raw_data[line_number].split(",")
        point_bundle_temp = []
        masks = line[1].replace("\n", "").split()
        for x in range(0, len(masks), 2):
            point_bundle_temp += [t for t in range(int(masks[x]), int(masks[x]) + int(masks[x + 1]))]
        for i in range(len(images)):
            if images[i][0] == line[0]:
                images[i][1] = images[i][1] + point_bundle_temp
                found = True
                break
        current_index += 1
        if current_index == minimized_index and minimized_index != 0:
            break
    with open(data_directory + 'hitmask', 'wb') as fp:
        pickle.dump(images, fp)
    return images


def extract_to_cnn(mask_size, step_size, reload=False):
    output_cnn = []
    if reload:
        images = get_image_data()
    else:
        with open(data_directory + 'hitmask', 'rb') as fp:
            images = pickle.load(fp)
    for i in images:
        if len(i[1]) == 0:
            zero_out = True
        else:
            zero_out = False
        temp = []
        for y in range(0, image_height - mask_size, step_size):
            for x in range(0, image_width - mask_size, step_size):
                if (x * (y + 1)) in i[1]:
                    temp.append(1)
                else:
                    temp.append(0)
        output_cnn.append(temp)
    with open(data_directory + 'masks\mask_' + str(mask_size) + '_' + str(step_size), 'wb') as fp:
        pickle.dump(output_cnn, fp)
