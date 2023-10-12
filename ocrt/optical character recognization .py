import tensorflow as tf
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# Dataset Preparation
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Model Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Perform OCR on a test image
test_image = x_test[0]
test_image = np.expand_dims(test_image, axis=0)
predicted_label = model.predict(test_image).argmax()
print('Predicted Label:', predicted_label)

# Load the sample text image
text_image = cv2.imread('/content/th.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image
desired_size = (28, 28)  # Use the same size as MNIST images
text_image = cv2.resize(text_image, desired_size)

# Preprocess the image
text_image = text_image / 255.0
text_image = np.expand_dims(text_image, axis=0)

# Perform OCR on the sample text image
predicted_label = model.predict(text_image).argmax()
print('OCR Predicted Label:', predicted_label)

# Display the image
cv2_imshow(text_image[0])

# Wait for key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving and Loading the Model
model.save("mnist_model.h5")  # Save the trained model

loaded_model = tf.keras.models.load_model("mnist_model.h5")  # Load the saved model

# Perform OCR using the loaded model on the sample text image
loaded_predicted_label = loaded_model.predict(text_image).argmax()
print('Loaded Model OCR Predicted Label:', loaded_predicted_label)

# Display the image
cv2_imshow(text_image[0])

# Wait for key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving and Loading the Model
model.save("mnist_model.h5")  # Save the trained model

loaded_model = tf.keras.models.load_model("mnist_model.h5")  # Load the saved model

# Perform OCR using the loaded model on the sample text image
loaded_predicted_label = loaded_model.predict(text_image).argmax()
print('Loaded Model OCR Predicted Label:', loaded_predicted_label)
