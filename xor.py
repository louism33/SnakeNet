import tensorflow as tf
import numpy as np

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_train = np.array([[0], [1], [1], [0]], dtype=float)
y_test = np.array([[0], [1], [1], [0]], dtype=float)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5000)
model.evaluate(x_test, y_test)


writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()
