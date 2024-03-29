import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import optimizers
import json

from Load_Data import DataGenerator_train
from Model_DLinkNet.D_LinkNet import D_LinkNet

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from segmentation_models.metrics import iou_score
from segmentation_models.losses import bce_dice_loss

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
    project_directory = r".\CelebAMask-HQ"

    return project_directory


# Set training hyperparameters 
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.08
LEARNING_RATE_DECAY_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 8
EPOCHS = 60

if __name__ == '__main__':
    # Load training data and divide into two parts: training set and testing set 
    split_ratio = 0.7
    with open(f'{get_project_dir()}\\train.txt', 'r') as file:
        data = file.read().split("\n")[:-1]
        random.shuffle(data)

    idx = int(np.floor(len(data) * (split_ratio)))
    train_list = data[0:idx]
    val_list = data[idx:-1]
    np.save('val_list.npy', np.array(val_list))

    # Load json file
    json_file_path = f"{get_project_dir()}\\data_dict.json"
    with open(json_file_path, 'r') as json_file:
        data_json = json.load(json_file)

    # Use DataGenerator to generate train batch and val batch
    train, train_count = DataGenerator_train(dir='train', img_list=train_list, data_json=data_json, IsAugmentation=True, batch_size=BATCH_SIZE)
    val, val_count  = DataGenerator_train(dir='val', img_list=val_list, data_json=data_json, IsAugmentation=False, batch_size=BATCH_SIZE)

    # Set callback function
    Reduce = ReduceLROnPlateau(monitor='val_loss',
                               factor=LEARNING_RATE_DECAY_FACTOR,
                               patience=LEARNING_RATE_DECAY_PATIENCE,
                               verbose=EARLY_STOPPING_PATIENCE,
                               mode='min')

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=EARLY_STOPPING_PATIENCE, 
                                   verbose=1, 
                                   mode='min') 

    csv_logger = CSVLogger('./training.log')

    filepath="DLinkNet-weights-improvement-{epoch:02d}-{val_loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [checkpoint, Reduce, early_stopping, csv_logger] 

    # Load model and set complie
    net_final = D_LinkNet(input_shape=(256,256,3), nclass=19, layers=[3,4,6,3])
    net_final.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE),
                      loss=bce_dice_loss,
                      metrics=iou_score)

    # Prepare dataset from train and val and calculate train/val step
    train_dataset = train.repeat()
    val_dataset = val.repeat()
    train_steps = train_count // BATCH_SIZE
    val_steps = val_count // BATCH_SIZE


    # Train model
    history = net_final.fit(train_dataset,
                            steps_per_epoch=train_steps,
                            validation_data=val_dataset,
                            validation_steps=val_steps,
                            epochs=EPOCHS,
                            callbacks=callbacks_list
                            )


    # Save acc/loss figure
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['iou_score'], label='iou_score')
    plt.plot(history.history['val_iou_score'], label='val_iou_score')
    plt.xlabel('Epoch')
    plt.ylabel('iou')
    # plt.ylim([0, 1])
    plt.legend(loc='lower right')

    # test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
    print('iou_score=',history.history['iou_score'][-1],"   ","val_iou_score=",history.history['val_iou_score'][-1])
    plt.savefig(r'.\DLinkNet_iou.png')
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    #plt.ylim([0.3, 1])
    plt.legend(loc='upper right')
    # test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
    print('loss=',history.history['loss'][-1],"   ","val_loss=",history.history['val_loss'][-1])
    plt.savefig(r'.\DLinkNet_loss.png')

tf.config.experimental.set_memory_growth(gpus[0],True)