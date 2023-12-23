from tensorflow.keras.layers import Conv2D, MaxPool2D, add, Dropout, Conv2DTranspose
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from .ResNet_Block import conv2d_bn, residual_block

def D_LinkNet(input_shape=(256,256,3), nclass=19, layers=[3,4,6,3]):

    input_ = Input(shape=input_shape)
    
    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = residual_block(64, layers[0], is_first_layer=True)(pool1)
    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    conv3 = residual_block(128, layers[1], is_first_layer=True)(pool2)
    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv3)
    
    conv4 = residual_block(256, layers[2], is_first_layer=True)(pool3)
    pool4 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv4)
         
    conv5 = residual_block(512, layers[3], is_first_layer=True)(pool4)
    
    conv6 = Conv2D(512, 3, dilation_rate=(1, 1), activation='relu', padding='same')(conv5)
    conv6 = Dropout(0.1)(conv6)
    conv7 = Conv2D(512, 3, dilation_rate=(2, 2), activation='relu', padding='same')(conv6)
    conv8 = Conv2D(512, 3, dilation_rate=(4, 4), activation='relu', padding='same')(conv7)
    conv9 = Conv2D(512, 3, dilation_rate=(8, 8), activation='relu', padding='same')(conv8)
    conv9_1 = Conv2D(512, 3, dilation_rate=(16, 16), activation='relu', padding='same')(conv9)
    conv9_1 = Dropout(0.2)(conv9_1)
    
    merge = add([conv5,conv6,conv7,conv8,conv9,conv9_1])
    
    #up-scaling
    conv10 = Conv2D(128, 1, activation='relu', padding='same')(merge)
    conv10 = Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same')(conv10)
    conv10 = Dropout(0.2)(conv10)
    conv10 = Conv2D(256, 1, activation='relu', padding='same')(conv10)
    merge1 = add([conv10,conv4])
    
    conv11 = Conv2D(64, 1, activation='relu', padding='same')(merge1)
    conv11 = Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, 1, activation='relu', padding='same')(conv11)
    merge2 = add([conv11,conv3])
    
    conv12 = Conv2D(32, 1, activation='relu', padding='same')(merge2)
    conv12 = Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='same')(conv12)
    conv12 = Dropout(0.1)(conv12)
    conv12 = Conv2D(64, 1, activation='relu', padding='same')(conv12)
    merge3 = add([conv12,conv2])
    
    conv13 = Conv2D(16, 1, activation='relu', padding='same')(merge3)
    conv13 = Conv2DTranspose(16, 3, strides=(2, 2), activation='relu', padding='same')(conv13)
    conv13 = Conv2D(64, 1, activation='relu', padding='same')(conv13)

    conv14 = Conv2DTranspose(32, 4, strides=(2, 2), activation='relu', padding='same')(conv13)
    output_ = Conv2D(nclass, 3, activation='sigmoid', padding='same')(conv14)
    
    model = Model(inputs=input_, outputs=output_)
    model.summary()

    return model
