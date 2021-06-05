import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil
import cv2
from keras.preprocessing import image                  
from tqdm.notebook import tqdm
from PIL import ImageFile                            

BASE_DIR = "/home/abhinav/distracted_driver/"

BASE_MODEL_PATH = os.path.join(BASE_DIR,"model")
PICKLE_DIR = os.path.join(BASE_DIR,"pickle_files")
JSON_DIR = os.path.join(BASE_DIR,"json_files")

if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

BEST_MODEL = os.path.join(BASE_MODEL_PATH,"self_trained","distracted-11-0.99.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # img = image.load_img(img_path, target_size=(128, 128))
    img = np.asarray(img_path)
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    # convert PIL.Image.Image type to 3D tensor with shape (128, 128, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 128, 128, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

# def paths_to_tensor(img_paths):
    # list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    # return np.vstack(list_of_tensors)

def return_prediction(filename):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    print(type(filename))
    test_tensors = path_to_tensor(filename).astype('float32')/255 - 0.5

    ypred_test = model.predict(test_tensors,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)

    print(ypred_class)
    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    print(id_labels)
    ypred_class = int(ypred_class)
    res = id_labels[ypred_class]

    # return class_name_result
    # creating the prediction results for the image classification and shifting the predicted images to another folder
    #with renamed filename having the class name predicted for that image using mode
    with open(os.path.join(os.getcwd(),'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
    prediction_result = info[res]
    return prediction_result

if __name__=='__main__':
    pass
