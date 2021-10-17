import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import json


""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Processing the data and saving the image and mask """
def process_data(image_path, json_path, save_dir):
    """ Reading the JSON file """
    i=0
    f = open(json_path, "r",encoding='UTF8')
    data = f.read()
    json_data=json.loads(data)
    json_img_data=json_data["_via_img_metadata"]
    """ Loop over the JSON data (dictionary) """
    for key, value in tqdm(json_img_data.items()):
        # print(key,value)
        filename = value["filename"]

        """ Extracting the name of the image, by removing its extension """
        input_name='input_%03d' % i
        label_name='label_%03d' % i
        # name = filename.split("_말_")[0]
        path= image_path
        img_name=filename
        full_path = path + '/' + img_name
        img_array=np.fromfile(full_path,np.uint8)
        img=cv2.imdecode(img_array,cv2.IMREAD_GRAYSCALE)
        H, W = img.shape

        """ Extracting information about the annotated regions """
        regions = value["regions"]

        if len(regions) == 0:
            mask = np.zeros((H, W))

        else:
            mask = np.zeros((H, W))

            for region in regions:
                cx = int(region["shape_attributes"]["cx"])
                cy = int(region["shape_attributes"]["cy"])
                rx = int(region["shape_attributes"]["r"])

                center_coordinates = (cx, cy)
                radius=rx
                color = (255, 255, 255)
                thickness = -1
                mask = cv2.circle(mask,center_coordinates,radius,color,thickness)

        """ Saving the image and mask """
        cv2.imwrite(f"{save_dir}/image/{input_name}.png", img)
        cv2.imwrite(f"{save_dir}/mask/{label_name}.png", mask)
        i+=1

if __name__ == "__main__":
    """ Dataset path """
    dataset_path = "7H-MGP912406000"
    dataset = glob(os.path.join(dataset_path, "*"))
    """ Loop over the dataset """
    for data in dataset:
        """ Path for the files """
        image_path = glob(os.path.join(data, "*"))[0]
        json_path = glob(os.path.join(image_path , "*.json"))[0]
        # print(data)
        """ Creating directories to save the data """
        dir_name = data.split("/")[0]
        save_dir = f"data/{dir_name}/bottom"
        create_dir(f"{save_dir}/image")
        create_dir(f"{save_dir}/mask")

        """ Process the data """
        process_data(image_path, json_path, save_dir)