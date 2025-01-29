import os
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def read_image(filepath: str, data_dir: str) -> np.ndarray:
    return cv2.imread(os.path.join(data_dir, filepath))


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(
        image.copy(), (image_size, image_size), interpolation=cv2.INTER_AREA
    )


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> Model:
    input_layer = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(
        input_layer
    )
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(input_layer, output_layer)

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.summary()
    return model


def main():
    base_dir = "."
    categories = ["COVID", "Non-COVID"]
    data_entries = []

    for category_id, category_name in enumerate(categories):
        for file in os.listdir(os.path.join(base_dir, category_name)):
            data_entries.append(
                (os.path.join(category_name, file), category_id, category_name)
            )

    data_df = pd.DataFrame(data_entries, columns=["Filepath", "CategoryID", "Category"])

    IMAGE_SIZE = 64
    images = np.zeros((data_df.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))

    for i, file in tqdm(enumerate(data_df["Filepath"].values)):
        image = read_image(file, base_dir)
        if image is not None:
            images[i] = resize_image(image, IMAGE_SIZE)

    X = images / 255.0
    y = data_df["CategoryID"].values
    y = to_categorical(y, num_classes=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.5, random_state=datetime.now().microsecond
    )

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    num_classes = 2
    model = build_model(input_shape, num_classes)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    learning_rate_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6
    )
    callbacks = [learning_rate_scheduler]

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.7, 1.3],
    )

    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    model_json = model.to_json()
    with open("ResNet50_Model.json", "w") as json_file:
        json_file.write(model_json)

    model.save("ResNet50_Model.keras")


if __name__ == "__main__":
    main()