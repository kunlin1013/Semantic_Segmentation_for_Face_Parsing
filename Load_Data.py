import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf
import numpy as np
import random

def DataGenerator_train(dir: str, img_list: list, data_json: dict, IsAugmentation: bool = True, batch_size: int = 32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = [256, 256]
    
    def load_and_preprocess_img(path, mask_path):
        def load_numpy_file(file_path):
            file_path = file_path.numpy()
            file_path_str = file_path.decode()
            numpy_array = np.load(file_path_str)
            return numpy_array.astype("float32")

        def load_mask(mask_path):
            mask_np = np.load(mask_path)
            return mask_np
        
        # import image
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        
        # import mask
        mask = tf.py_function(load_numpy_file, [mask_path], Tout=tf.float32)
        mask.set_shape([256, 256, 19])
        
        return image, mask
    
    def activator(images, augmenter, parents, default):
        return False if augmenter.name in ["UnnamedAddToHueAndSaturation", "UnnamedGaussianBlur", "UnnamedGaussianNoise", "UnnamedSpeckleNoise", "UnnamedRemoveSaturation"] else default
    
    def augmentation(img, mask):
        def sequential_aug(img1, img2):
            img1 = tf.cast(img1 * 255, tf.uint8)
            img2 = tf.cast(img2 * 255, tf.uint8)
            
            sometimes = lambda aug: iaa.Sometimes(0.3, aug)  # apply operations on 50% of input data
            seq = iaa.Sequential([sometimes(iaa.SomeOf(1, 
                                                        [iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                                                                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                                                    rotate=(-30, 30), order=3, cval=0),
                                                        iaa.PerspectiveTransform(scale=(0.01, 0.15))]
                                                        )
                                            ),
                                   sometimes(iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
                                   iaa.Flipud(0.3), # horizontally flip 30% of the images
                                   sometimes(iaa.SomeOf(1,
                                                        [iaa.GaussianBlur(sigma=(0.5, 3.0)),
                                                         iaa.imgcorruptlike.GaussianNoise(severity=2),
                                                         iaa.imgcorruptlike.SpeckleNoise(severity=2)]
                                                        )
                                              ),
                                   iaa.Sometimes(0.1, iaa.RemoveSaturation(1.0))
                               ])
            
            seq = seq.to_deterministic()
            img1 = seq.augment_image(img1.numpy())
            img2 = seq.augment_image(img2.numpy(), hooks=ia.HooksImages(activator=activator))
            
            return img1, img2
        
        img, mask = tf.py_function(sequential_aug, [img, mask], (tf.float32, tf.float32))
        img.set_shape([256, 256, 3])
        mask.set_shape([256, 256, 19])
        
        img = tf.cast(img, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32) / 255.0
        
        return img, mask
    
    # Get path to all files
    img_path_list = []
    mask_list = []
    
    for img_idx in img_list:
        img_path_list.append(data_json[str(img_idx)]['filepath'])
        mask_list.append(data_json[str(img_idx)]['npy_path'])
            
    print('Total samples:', len(mask_list))
    
    # Shuffle the list
    temp = list(zip(img_path_list, mask_list))
    random.shuffle(temp)
    img_path_list, mask_list = zip(*temp)
    img_path_list, mask_list = list(img_path_list), list(mask_list)
    
    X = img_path_list
    Y = mask_list
    
    # Construct tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    
    def load_dataset(x, y):
        return tf.data.Dataset.from_tensors(load_and_preprocess_img(x, y))
    data = data.interleave(load_dataset, cycle_length=7, num_parallel_calls=AUTOTUNE)
    # data = data.map(load_and_preprocess_img, AUTOTUNE)
#     data = data.cache()
    if dir == "train" and IsAugmentation:
        data = data.map(augmentation, AUTOTUNE) # augment only the training dataset
    
    # Add all the settings
    data = data.shuffle(1024)
    data = data.batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    
    total_data = len(img_path_list)
    
    return data, total_data

def DataGenerator_test(dir: str, img_list: list, data_json: dict, IsAugmentation: bool = True, batch_size: int = 32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = [256, 256]
    
    def load_and_preprocess_img(path, mask_path):
        def load_numpy_file(file_path):
            file_path = file_path.numpy()
            file_path_str = file_path.decode()
            numpy_array = np.load(file_path_str)
            return numpy_array.astype("float32")

        # import image
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        
        # import mask
        mask = tf.py_function(load_numpy_file, [mask_path], Tout=tf.float32)
        mask.set_shape([256, 256, 19])
        
        return image, mask
    
    # Get path to all files
    img_path_list = []
    mask_list = []
    
    for img_idx in img_list:
        img_path_list.append(data_json[str(img_idx)]['filepath'])
        mask_list.append(data_json[str(img_idx)]['npy_path'])
            
    print('Total samples:', len(mask_list))
    
    # Shuffle the list
    temp = list(zip(img_path_list, mask_list))
    img_path_list, mask_list = zip(*temp)
    img_path_list, mask_list = list(img_path_list), list(mask_list)
    
    X = img_path_list
    Y = mask_list
    
    # Construct tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices((X, Y))
    data = data.map(load_and_preprocess_img, AUTOTUNE)
#     data = data.cache()
    
    # Add all the settings
    data = data.batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    
    total_data = len(img_path_list)
    
    return data, total_data

def DataGenerator_unseen(dir: str, img_path_list: list, IsAugmentation: bool = True, batch_size: int = 32):
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = [256, 256]
    
    def load_and_preprocess_img(path):
        # import image
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
    
        return image
    
    # Get path to all files
    print('Total samples:', len(img_path_list))
    
    # Shuffle the list
    X = img_path_list
    
    # Construct tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices(X)
    data = data.map(load_and_preprocess_img, AUTOTUNE)
#     data = data.cache()
    
    # Add all the settings
    data = data.batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    
    total_data = len(img_path_list)
    
    return data, total_data