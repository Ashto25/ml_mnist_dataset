import wandb
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt



# Load and process the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Split the data into test and train sets
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)



# Create the model using sequential layers
# Layer 1 - convolutional 2D layer to take the image as input
# Layer2 - Fully connected dense layer with 64 nodes using a relu activation function
# Layer3 - Fully connected dense layer which has 10 nodes as the output for numbers 0-9
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="handwriting_test",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# Fit the model to the data through 5 epochs of batch sizes 64. 
# The validation split of 0.2 allows for 20% of the data to be used as validation
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
# Evaluate the model using the images and labels set aside
test_loss, test_acc = model.evaluate(test_images, test_labels)

#Printout the accuracy of the tests
print(f"Test accuracy: {test_acc}")



# Log metrics to WandB
for epoch in range(5):
    wandb.log({"train_loss": history.history['loss'][epoch],
               "train_accuracy": history.history['accuracy'][epoch],
               "val_loss": history.history['val_loss'][epoch],
               "val_accuracy": history.history['val_accuracy'][epoch]})

wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

# Finish to wandb
wandb.finish()



# Create a visualization using matplotlib to show images along with their predictions
num_images_to_visualize = 5
test_predictions = model.predict(test_images[:num_images_to_visualize])
test_predictions_labels = [f"Predicted: {str(label.argmax())}" for label in test_predictions]
plt.figure(figsize=(10, 6))
for i in range(num_images_to_visualize):
    plt.subplot(1, num_images_to_visualize, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
    plt.title(test_predictions_labels[i])
    plt.axis("off")

plt.tight_layout()
plt.show()