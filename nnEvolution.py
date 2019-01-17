import keras
import random

save_best_model = True
save_all = False
total_models = 25
a = keras.layers.Conv2D(3, kernel_size=5, input_shape=(28, 28, 1), activation='relu')

# if train_previous_model:
#     json_file = open('fashion_model_v1.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = keras.models.model_from_json(loaded_model_json)
#     model.load_weights("fashion_model_v1.h5")
#     print("Loaded model from disk")

models = []
for i in range(total_models):
    models.append(keras.Sequential())

    kernel_size_1 = random.randint(2, 4) * 2 + 1
    kernel_size_2 = random.randint(2, 4) * 2 + 1
    hidden_neurons_number = random.randint(100, 1000)
    dropout_number = random.random()

    model = models[i]
    model.add(keras.layers.Conv2D(3, kernel_size=kernel_size_1, input_shape=(28, 28, 1), activation='relu'))
    model.add(keras.layers.Conv2D(3, kernel_size=kernel_size_2, activation='relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(dropout_number))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hidden_neurons_number, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

scores = []
for i in range(len(models)):
    model = models[i]
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10, verbose=2, validation_data=(x_test, y_test), shuffle=True)

    x_test = x_test.reshape(10000, 28, 28, 1)
    score = model.evaluate(x_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    scores.append(score[1] * 100)
    print("scores so far:")
    print(scores)
    print("Best net so far:")
    print(scores.index(max(scores)))

if save_best_model:
    best_model = models[scores.index(max(scores))]
    name = "best_model_1"
    model_json = best_model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
        best_model.save_weights(name + ".h5")
    print("Saved best model to disk")

for i in range(len(models)):
    model = models[i]
    name = "fashion_model_" + str(i)

    if save_all:
        model_json = model.to_json()
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(name+".h5")
        print("Saved model" + str(i) +" to disk")


print()
print("final scores are:")
print(scores)