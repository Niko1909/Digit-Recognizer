# imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# load data
data = pd.read_csv('train.csv')
X = data.drop('label', axis=1)
X = tf.convert_to_tensor(X)

y = data.label
y = tf.convert_to_tensor(y)

# split data
X_train, X_valid = X[:30000], X[30000:]
y_train, y_valid = y[:30000], y[30000:]

# reshape data
X_train = tf.reshape(X_train, [30000, 28, 28, 1])
X_valid = tf.reshape(X_valid, [12000, 28, 28, 1])

# iteration 1: basic model
def build_model1(X_train, X_valid, y_train, y_valid):
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(x=X_train, y=y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=1)
    history_frame = pd.DataFrame(history.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()
    return model, history_frame

model1, history = build_model1(X_train, X_valid, y_train, y_valid) 

# iteration 2: basic model with max pooling - improved training time greatly and accuracy slightly
def build_model2(X_train, X_valid, y_train, y_valid):
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]),
        layers.MaxPooling2D(padding='same'),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Flatten(),
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(x=X_train, y=y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=1)
    history_frame = pd.DataFrame(history.history)
    return model, history_frame

model2, history = build_model2(X_train, X_valid, y_train, y_valid) 

# iteration 3: builds on iteration 2 with batch normalization and dropout - improved accuracy (prevents overfitting)
def build_model3(X_train, X_valid, y_train, y_valid):
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[28, 28, 1]),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(units=64, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x=X_train, y=y_train, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[callback])
    history_frame = pd.DataFrame(history.history)
    return model, history_frame

model3, history = build_model3(X_train, X_valid, y_train, y_valid) 

# iteration 4: build on iteration 3 with data augmentation
def build_model4(X_train, X_valid, y_train, y_valid):
    data_aug = keras.Sequential([
        layers.Rescaling(scale=1./255),
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomRotation(factor=(-0.1, 0.1)),
        layers.RandomContrast(factor=(0.1, 0.2))
    ])

    model = keras.Sequential([
        data_aug,
        layers.BatchNormalization(input_shape=[28, 28, 1]),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(units=64, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x=X_train, y=y_train, epochs=15, validation_data=(X_valid, y_valid), verbose=1, callbacks=[callback])
    history_frame = pd.DataFrame(history.history)
    return model, history_frame

model4, history = build_model4(X_train, X_valid, y_train, y_valid) 

# iteration 5: pretrained base with trainable head using the augmentation from iteration 4
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

def create_dataset(X_train, y_train, X_valid, y_valid):
    data_aug = keras.Sequential([
        layers.Rescaling(scale=1./255),
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomRotation(factor=(-0.1, 0.1)),
        layers.RandomContrast(factor=(0.1, 0.2))
    ])
    # pass data through data augmentation
    X_train = data_aug(X_train)
    X_valid = data_aug(X_valid)
    # convert to float
    X_train = tf.image.convert_image_dtype(X_train, dtype=tf.float32)
    X_valid = tf.image.convert_image_dtype(X_valid, dtype=tf.float32)

    # create dataset and reshape data
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    train_ds = train_ds.map(preprocess_image)
    valid_ds = valid_ds.map(preprocess_image)

    return train_ds, valid_ds

# resize and colourize data for pretrained model
def preprocess_image(images, labels):
    images = tf.image.resize(images, (224, 224))
    images = tf.image.grayscale_to_rgb(images)
    return images, labels

def build_model5(train_ds, valid_ds):
    # add pretrained base model
    base = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    base.trainable = False
    
    model = keras.Sequential([
        base,
        layers.Flatten(),
        layers.Dense(units=64, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=1,
        verbose=0,
    )
    history_frame = pd.DataFrame(history.history)
    return model, history_frame

train_ds, valid_ds = create_dataset(X_train, y_train, X_valid, y_valid)

# ==============CAREFUL: building this model uses a lot of memory (20gb+) and can take hours if using a CPU to train=================

# model5, history = build_model5(train_ds, valid_ds) 

# creation submissions
# testing
test_data = pd.read_csv('test.csv')
X_test = tf.reshape(test_data, [test_data.shape[0], 28, 28, 1])

def make_preds(model, X_test):
    preds = model.predict(X_test)
    preds = [tf.argmax(pred).numpy() for pred in preds]
    label = [i for i in range(1, len(preds)+1)]
    return pd.concat([pd.Series(label), pd.Series(preds)], axis=1, keys=['ImageId', 'Label'])

submission1 = make_preds(model1, X_test)
submission1.to_csv('submission1.csv', index=False)

submission2 = make_preds(model2, X_test)
submission2.to_csv('submission2.csv', index=False)

submission3 = make_preds(model3, X_test)
submission3.to_csv('submission3.csv', index=False)

submission4 = make_preds(model4, X_test)
submission4.to_csv('submission4.csv', index=False)