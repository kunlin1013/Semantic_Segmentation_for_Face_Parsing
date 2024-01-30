import numpy as np
import tensorflow as tf
import json
from CSV_process.csv_utils import array2csv

from segmentation_models.metrics import iou_score
from segmentation_models.losses import  bce_dice_loss
from Load_Data import DataGenerator_test
from thresholding import convert_binary_adaptive_thres
from thresholding import convert_binary

labels_celeb = ['background','skin','nose',
                    'eye_g','l_eye','r_eye','l_brow',
                    'r_brow','l_ear','r_ear','mouth',
                    'u_lip','l_lip','hair','hat',
                    'ear_r','neck_l','neck','cloth']

custom_objects = {"binary_crossentropy_plus_dice_loss" : bce_dice_loss,
                  "iou_score" : iou_score
                  }

BATCH_SIZE = 16
NUM_IMAGE_ITER = 300

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def get_project_dir():
    # Get the full path to the repository
    return r".\CelebAMask-HQ"

if __name__ == '__main__':

    # Load testing data
    with open(f'{get_project_dir()}\\test.txt', 'r') as file:
        data = file.read().split("\n")[:-1]
    test_list = data

    # Load json file
    json_file_path = f"{get_project_dir()}\\data_dict.json"
    with open(json_file_path, 'r') as json_file:
        data_json = json.load(json_file)
    

    # Load trained model
    net_final1 = tf.keras.models.load_model('.\Weight\DLinkNet\8809-31-0.252.h5',custom_objects=custom_objects)
    net_final2 = tf.keras.models.load_model('.\Weight\DLinkNet\8808-44-0.261.h5',custom_objects=custom_objects)
    net_final3 = tf.keras.models.load_model('.\Weight\DLinkNet\8791-33-0.267.h5',custom_objects=custom_objects)
    net_final4 = tf.keras.models.load_model('.\Weight\DLinkNet\8781-59-0.281.h5',custom_objects=custom_objects)
    net_final5 = tf.keras.models.load_model('.\Weight\DLinkNet\8768-47-0.258.h5',custom_objects=custom_objects)

    # Iteratively predict masks
    num_total_image = len(test_list)
    num_iter = int(np.ceil(num_total_image / NUM_IMAGE_ITER))
    is_first_row = True

    for it in range(num_iter):
        starting_image_id = it * NUM_IMAGE_ITER
    
        # Fetch test data and predict result
        test, test_count = DataGenerator_test(dir='test',
                                              img_list=test_list[starting_image_id:np.min((starting_image_id+NUM_IMAGE_ITER, num_total_image))],
                                              data_json=data_json,
                                              IsAugmentation=False,
                                              batch_size=BATCH_SIZE)
        predict_test1 = net_final1.predict(test, verbose=1)
        predict_test2 = net_final2.predict(test, verbose=1)
        predict_test3 = net_final3.predict(test, verbose=1)
        predict_test4 = net_final4.predict(test, verbose=1)
        predict_test5 = net_final5.predict(test, verbose=1)
        
        all_predict = (predict_test1 + predict_test2 + predict_test3 + predict_test4 + predict_test5) / 5
    
        # Threshold using fixed value
        predict_result = convert_binary(all_predict, 0.7)
    
        # Set background to zero
        predict_result[:, :, :, 0] = np.zeros(predict_result.shape[:3])
    
        # Convert prediction to .csv fiile
        array2csv(predict_result, starting_image_id=starting_image_id, header_needed=is_first_row)
        is_first_row = False