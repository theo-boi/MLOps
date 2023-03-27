import os
import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlflow
import mlflow.keras

import tensorflow
import tensorflow.keras

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="path to train data")
    parser.add_argument("--test", type=str, help="path to test data")
    parser.add_argument("--learning_rate", required=False, default=0.001, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.tensorflow.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("train data:", args.train)
    print("train data:", args.test)

    # create the training & test sets, skipping the header row with [1:]
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # Extracting the label column
    X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
    y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
    
    X_test = test.values.astype('float32')

    # Convert train datset to (num_images, img_rows, img_cols, colour_channel_gray) format
    X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

    mlflow.log_metric("num_features", X_train.shape[1]*X_train.shape[2])

    mean_px = X_train.mean().astype(np.float32)
    std_px = X_train.std().astype(np.float32)

    def standardize(x): 
        return (x-mean_px)/std_px

    y_train= to_categorical(y_train)
    
    mlflow.log_metric("num_samples", y_train.shape[1])
    ####################
    #</prepare the data>
    ####################

    ##################
    #<train the model>
    ##################
    # fix random seed for reproducibility
    seed = 43
    np.random.seed(seed)

    batch_size = 64
    gen = ImageDataGenerator()
    batches = gen.flow(X_train, y_train, batch_size=batch_size)

    def get_bn_model():
        model = Sequential([
            Lambda(standardize, input_shape=(28,28,1)),
            Convolution2D(32,(3,3), activation='relu'),
            BatchNormalization(axis=1),
            Convolution2D(32,(3,3), activation='relu'),
            MaxPooling2D(),
            BatchNormalization(axis=1),
            Convolution2D(64,(3,3), activation='relu'),
            BatchNormalization(axis=1),
            Convolution2D(64,(3,3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dense(10, activation='softmax')
            ])
        model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    model = get_bn_model()
    model.optimizer.lr = args.learning_rate

    print(f"Training with data of shape {X_train.shape}")

    model.fit(batches, steps_per_epoch=len(batches), epochs=5)
    ###################
    #</train the model>
    ###################
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
