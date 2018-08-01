#IMPORTS
import os
import cv2

#GLOBALS
data_directory="D:\\Projects\\kaggle_cascade\\Data\\"
image_width=768
image_height=768

def get_image_data():
    images=[]
    minimized_index=100
    current_index=0
    for filename in os.listdir(data_directory+"train\\"):
        images.append([filename,cv2.imread(data_directory+"train\\"+filename)])
        current_index+=1
        if current_index==minimized_index and minimized_index!=0:
            break
    image_data=[]
    raw_data=open(data_directory+"train_ship_segmentations.csv","r").readlines()
    current_index=0
    for line_number in range(1,len(raw_data)):
        line=raw_data[line_number].split(",")
        point_bundle_temp=[]
        masks=line[1].replace("\n","").split()
        for x in range(0,len(masks),2):
            point_bundle_temp+=[t for t in range(int(masks[x]),int(masks[x])+int(masks[x+1]))]
        image_data.append([line[0],point_bundle_temp])
        current_index+=1
        if current_index==minimized_index and minimized_index!=0:
            break
    return image_data
#def extract_to_cnn(mask_size,step_size):
