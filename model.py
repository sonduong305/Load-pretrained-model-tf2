import numpy as np

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

mobile = tensorflow.keras.applications.mobilenet.MobileNet()

def MobileNetModel():
    # CREATE THE MODEL ARCHITECTURE

    # Exclude the last 5 layers of the above model.
    # This will include all layers up to and including global_average_pooling2d_1
    x = mobile.layers[-6].output

    # Create a new dense layer for predictions
    # 7 corresponds to the number of classes
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)

    # inputs=mobile.input selects the input layer, outputs=predictions refers to the
    # dense layer we created above.

    model = Model(inputs=mobile.input, outputs=predictions)

    # model.summary()
    model.load_weights("D:\\ws\\Skin cancer\\models\\model_mau.h5")

    return model