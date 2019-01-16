import tensorflow as tf
from tensorflow.keras.models import model_from_json

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

json_file = open('model_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_v1.h5")
print("Loaded model from disk")


loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# x_test = x_test.reshape(10000, 28, 28, 1)
# score = loaded_model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
