import numpy as np
from keras.datasets import mnist
from tkinter import Tk, Canvas, Button
from PIL import Image, ImageDraw, ImageOps
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.biases1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.biases2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    # Activation function, if score is 0 or less return 0, else return the input
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    # Convert logits- output scores to probabilities
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Cross entropy loss
    def compute_loss(self, y_true, y_pred):
        num_samples = y_true.shape[0]
        #1e-15 so the algorythm wont calculate the log of 0
        log_preds = np.log(y_pred + 1e-15)
        loss = -np.sum(y_true * log_preds) / num_samples
        return loss

    def forward(self, X):
        # Input to hidden layer, z1=x*w1+b1
        self.z1 = np.dot(X, self.weights1) + self.biases1
        # Hidden layer activation, applies relu function
        self.a1 = self.relu(self.z1)
        # Hidden layer to output, z2=a1*w2+b2
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        #Output layer activation, applies softmax
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        num_samples = X.shape[0]
        error_output = y_pred - y_true
        #Gradients for the output layer, dw2 - gradient loss weights, db2 - gradient loss biases
        #Stochastic gradient descent
        dW2 = np.dot(self.a1.T, error_output) / num_samples
        db2 = np.sum(error_output, axis=0, keepdims=True) / num_samples

        #Hidden layer error
        error_hidden = np.dot(error_output, self.weights2.T) * self.relu_derivative(self.z1)

        #Gradients for the hidden layer
        # The same as dw2 and db1
        dW1 = np.dot(X.T, error_hidden) / num_samples
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / num_samples

        self.weights2 -= self.learning_rate * dW2
        self.biases2 -= self.learning_rate * db2
        self.weights1 -= self.learning_rate * dW1
        self.biases1 -= self.learning_rate * db1

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            if epoch % 10 == 0:
                loss = self.compute_loss(y, y_pred)
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)


def draw_digit(mlp):
    root = Tk()
    root.title("Draw a digit")

    canvas_width = 280
    canvas_height = 280
    canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack()

    image = Image.new('L', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(image)

    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)
        draw.line([x1, y1, x2, y2], fill='black', width=20)

    canvas.bind('<B1-Motion>', paint)

    def preprocess_and_predict():
        resized_image = image.resize((28, 28))
        inverted_image = ImageOps.invert(resized_image)
        grayscale_image = inverted_image.convert('L')
        digit_array = np.array(grayscale_image).astype('float32')
        digit_array = np.where(digit_array > 128, 1.0, 0.0)
        plt.imshow(grayscale_image, cmap='gray')
        plt.title("Processed Image")
        plt.show()
        digit_array = digit_array.reshape(1, -1)
        prediction = mlp.predict(digit_array)
        print(f'Predicted digit: {prediction[0]}')
        root.destroy()

    button = Button(root, text='Predict', command=preprocess_and_predict)
    button.pack()

    root.mainloop()


def draw_digit_from_image(mlp, image_path):
    image = Image.open(image_path)
    grayscale_image = image.convert('L')
    resized_image = grayscale_image.resize((28, 28))
    digit_array = np.array(resized_image).astype('float32')
    if np.mean(digit_array) > 128:
        digit_array = 255 - digit_array

    plt.imshow(resized_image, cmap='gray')
    plt.title("Processed Image")
    plt.show()

    digit_array = np.where(digit_array > 128, 1.0, 0.0)
    digit_array = digit_array.flatten()
    digit_array = digit_array.reshape(1, -1)

    prediction = mlp.predict(digit_array)
    print(f'Predicted digit: {prediction[0]}')


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('Confusion Matrix.png')


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)

    train_images = np.where(train_images > 128, 255, 0).astype('float32') / 255.0
    test_images = np.where(test_images > 128, 255, 0).astype('float32') / 255.0

    #One hot encoding format
    train_labels_one_hot = np.zeros((train_labels.size, train_labels.max() + 1))
    train_labels_one_hot[np.arange(train_labels.size), train_labels] = 1


    test_labels_one_hot = np.zeros((test_labels.size, test_labels.max() + 1))
    test_labels_one_hot[np.arange(test_labels.size), test_labels] = 1

    input_size = train_images.shape[1]
    hidden_size = 128
    output_size = train_labels_one_hot.shape[1]
    mlp = MLP(input_size, hidden_size, output_size, learning_rate=0.1)

    weights_file = 'weights.npz'
    if os.path.exists(weights_file):
        pretrained_weights = np.load(weights_file)
        mlp.weights1 = pretrained_weights['weights1']
        mlp.biases1 = pretrained_weights['biases1']
        mlp.weights2 = pretrained_weights['weights2']
        mlp.biases2 = pretrained_weights['biases2']

    else:
        mlp.train(train_images, train_labels_one_hot, epochs=10000)

        np.savez(weights_file, weights1=mlp.weights1, biases1=mlp.biases1,
                 weights2=mlp.weights2, biases2=mlp.biases2)

    accuracy = mlp.evaluate(test_images, test_labels_one_hot)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    y_pred = mlp.predict(test_images)
    y_true = test_labels
    plot_confusion_matrix(y_true, y_pred, classes=range(10))

    while True:
        choice = input("Choose an option:\n1. Draw a digit\n2. Import an image\nEnter your choice (1 or 2): ").strip()

        if choice == '1':
            draw_digit(mlp)
        elif choice == '2':
            image_path = input("Enter the path to the image file: ").strip()
            draw_digit_from_image(mlp, image_path)
        else:
            print("Invalid choice. Please enter either 1 or 2.")

        another = input("Do you want to try again? Press y to retry or any other key to end: ").strip().lower()
        if another != 'y':
            break

if __name__ == "__main__":
    main()