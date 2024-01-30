import numpy as np
import cv2

def convert_binary(image_matrix, thresh_val=0.9):
    # Pxiel values for white and black
    white = 255
    black = 0
    
    # Determine if the value larger than the threshold and set as white pixel
    result = np.where(image_matrix >= thresh_val, white, black).astype(np.uint8)
    
    return result

def convert_binary_otsu(image_matrix):
    # Pxiel values for white and black
    white = 255
    black = 0
    
    # Determine if the value larger than the threshold and set as white pixel
    result = (image_matrix.copy() * 255.0).astype(np.uint8)
    for id in range(image_matrix.shape[0]):
        for mask_id in range(image_matrix.shape[3]):
            _, result[id, :, :, mask_id] = cv2.threshold(result[id, :, :, mask_id], black, white, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return result

def convert_binary_adaptive_thres(image_matrix, thresh_list):
    # Pxiel values for white and black
    white = 255
    black = 0
    
    # Determine if the value larger than the threshold and set as white pixel
    result = np.empty(image_matrix.shape, dtype=np.uint8)
    for channel in range(image_matrix.shape[-1]):
       threshold = thresh_list[channel]
       result[:, :, :, channel] = np.where(image_matrix[:, :, :, channel] >= threshold, white, black)
    
    return result