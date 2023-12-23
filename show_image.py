import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def plot_image_from_batch(img_batch, mask_batch):
    batch_size = img_batch.shape[0]
    num_cols = int(math.sqrt(batch_size))
    num_rows = math.ceil(batch_size / num_cols)
    
    fig, axes = plt.subplots(num_rows, 2*num_cols, figsize=(15, 15), gridspec_kw={'hspace': 0.005})
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        
        axes[row, 2*col].imshow((img_batch[i]))
        axes[row, 2*col].axis('off')
        axes[row, 2*col+1].imshow(mask_batch[i,:,:,0])
        axes[row, 2*col+1].axis('off')
    
    plt.show()
    
def plot_image_from_npy(img_path, mask_path):
    
    # load image and mask
    img = np.asarray(Image.open(img_path))
    mask = np.load(mask_path)
    
    # plot
    fig, axes = plt.subplots(5, 4, figsize=(20, 20), gridspec_kw={'hspace': 0.005})
    axes[0, 0].imshow(img)
    axes[0, 0].axis('off')
    for i in range(1, mask.shape[2] + 1):
        row = i // 4
        col = i % 4
        axes[row, col].imshow((mask[:,:,i-1]), cmap='gray')
        axes[row, col].axis('off')
    plt.show()

def plot_result(img_path, predict_mask):
    labels_celeb = ['background','skin','nose',
                    'eye_g','l_eye','r_eye','l_brow',
                    'r_brow','l_ear','r_ear','mouth',
                    'u_lip','l_lip','hair','hat',
                    'ear_r','neck_l','neck','cloth']
    
    img = np.asarray(Image.open(img_path))
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 20), gridspec_kw={'hspace': 0.005})
    axes[0, 0].imshow(img)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Originial Image')
    for i in range(1, predict_mask.shape[2] + 1):
        row = i // 4
        col = i % 4
        axes[row, col].imshow((predict_mask[:,:,i-1]), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"{labels_celeb[i-1].capitalize()} mask")
    plt.show()