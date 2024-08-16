import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load the trained model
new_model = tf.keras.models.load_model('digit_classifier.keras', custom_objects={'softmax_v2': tf.keras.activations.softmax})

# Define the drawing canvas
class DrawingCanvas:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = self.fig.canvas
        self.drawing = False
        self.xs = []
        self.ys = []
        self.ax.set_xlim(0, 28)
        self.ax.set_ylim(0, 28)
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.btn = Button(plt.axes([0.4, 0.01, 0.2, 0.075]), 'Predict')
        self.btn.on_clicked(self.predict)

    def on_press(self, event):
        self.drawing = True
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        
    def on_release(self, event):
        self.drawing = False
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.draw()
        
    def on_motion(self, event):
        if self.drawing:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.draw()
    
    def draw(self):
        self.ax.plot(self.xs, self.ys, color='black')
        self.canvas.draw()

    def predict(self, event):
        self.ax.clear()
        self.ax.set_xlim(0, 28)
        self.ax.set_ylim(0, 28)
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Create an image from the drawing
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = img[:, :, 0]  # Convert to grayscale
        img = np.invert(img)  # Invert colors
        img = tf.image.resize(img, [28, 28])  # Resize to 28x28
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict
        prediction = new_model.predict(img)
        predicted_digit = np.argmax(prediction)
        print(f"Predicted Digit: {predicted_digit}")

        # Display the image
        self.ax.imshow(img[0], cmap='gray')
        plt.title(f"Predicted Digit: {predicted_digit}")
        self.canvas.draw()

# Create and display the drawing canvas
drawing_canvas = DrawingCanvas()
plt.show()
