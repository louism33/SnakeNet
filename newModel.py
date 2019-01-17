import tensorflow as tf
import keras

train_previous_model = True
save_model = True

if train_previous_model:
    json_file = open('fashion_model_v1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights("fashion_model_v1.h5")
    print("Loaded model from disk")

else:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(3, kernel_size=5, input_shape=(28, 28, 1), activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(3, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

model.fit(x=x_train, y=y_train, epochs=15, verbose=1, validation_data=(x_test, y_test), shuffle=True)

if save_model:
    model_json = model.to_json()
    with open("fashion_model_v1.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("fashion_model_v1.h5")
    print("Saved model to disk")