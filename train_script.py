import numpy as np
from tensorflow import keras

"""
https://keras.io/examples/vision/mnist_convnet/
"""

def preprocess(input_data):
        
    # Scale images to the [0, 1] range
    preprocessed_data = input_data.astype("float32") / 255

    # Data has shape (n, height, width), add a channel dimension so it becomes (n, height, width, 1)
    preprocessed_data = np.expand_dims(preprocessed_data, axis=-1)

    return preprocessed_data


def construct_cnn_model(input_shape, num_classes):

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def train_cnn_model(X, Y):

    image_dims = x_train.shape[1:]

    model = construct_cnn_model(
        input_shape=image_dims,
        num_classes=num_classes
    )

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 128
    epochs = 10

    model.fit(
        X,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=2
    )

    return model


def evaluate(model, X, Y):

    score = model.evaluate(X, Y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':

    # Load data 
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    num_classes = 10
    num_train_samples = x_train.shape[0]
    num_test_samples = x_train.shape[0]
    print(f'Loaded {num_train_samples} training samples, {num_test_samples} testing samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Preprocess
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)


    # Train
    model = train_cnn_model(x_train, y_train)

    # Save the model
    model.save('cnn-model-v1')

    # Evaluate
    evaluate(model, x_test, y_test)


