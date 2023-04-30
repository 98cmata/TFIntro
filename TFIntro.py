import tensorflow as tf
import numpy as np

# EXAMPLE ONE: LINEAR REGRESSION
'''
# Define the input data
x_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 6, 9, 11]   # would be slope of 2 but changed where a 7 would be to a 6

# Define the model parameters
W = tf.Variable(2.0)   # weights
b = tf.Variable(1.0)   # biases


# Define the model function
def linear_regression(x):
    return W * x + b


# Define the loss function
def mean_square_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Define the optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Train the model
for i in range(1000):
    with tf.GradientTape() as tape:
        y_pred = linear_regression(x_data)
        loss = mean_square_error(y_pred, y_data)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

# Predict the output for a new input
new_x = 0
new_y = linear_regression(new_x)
print(new_y.numpy())    # our prediction

'''


# EXAMPLE TWO: FEEDFORWARD NEURAL NETWORK MODEL
'''
# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Dense(3, input_dim=2, activation='sigmoid'),
  tf.keras.layers.Dense(2, activation='sigmoid'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
'''


# EXAMPLE 3 MINI FASHION MNIST (FOLLOW ALONG)

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize the data, 255 is a grayscale value
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  #tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)


# EXAMPLE 4 MNIST CONVOLUTIONAL NEURAL NETWORK
'''
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape the images to 4D arrays
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Define the input data
input_data = test_images[:5]
input_label = test_labels[:5]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Make a prediction
prediction = model.predict(input_data)
print("Predicted first image:", np.argmax(prediction[0]), "Actual label:", input_label[0])

#print("Predicted second image:", np.argmax(prediction[1]), "Actual label:", input_label[1])
#print("Predicted third image:", np.argmax(prediction[2]), "Actual label:", input_label[2])
#print("Predicted fourth image:", np.argmax(prediction[3]), "Actual label:", input_label[3])
#print("Predicted fifth image:", np.argmax(prediction[4]), "Actual label:", input_label[4])
'''

# DOWNLOAD IF ERROR: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

