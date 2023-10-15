 #import the necessary libraries
import tkinter as tk
from tkinter import filedialog,messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa

#function to process the audio file
def process_audio(audio_path):
    audio, fs = librosa.load(audio_path)
    D = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    # Save the spectrogram
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='linear',cmap='viridis')
    plt.savefig('output.png', dpi=300, bbox_inches='tight', pad_inches=0,transparent=True)

def predict():
    messagebox.showinfo("Processing Result","Processing")                    #show dialog box to indicate start of processing
    img = cv.imread("output.png")                                            #read image
    img = cv.resize(img,(300,300))                                           #resize image 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)                                 #convert images to gray
    img = np.expand_dims(img,0)                                              #expand image dimension to match that of the input layer to the NN
    img = np.expand_dims(img,-1)
    nm = tf.keras.models.load_model('my_model.h5')                           #Load tensorflow model
    bb = nm.predict(img)                                                     #predict model
    bb = int(bb[0][0])                                                       #get index of the predicted class
    # print(bb)f
    if bb == 0:
        messagebox.showinfo("Processing Result Press ok to Continue", "Voice Does not Match!")
    else:
        messagebox.showinfo("Processing Result Press ok to Continue", "Voice Matched")

#function to load audio file and get prediction
def browse_audio_file():
    audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
    if audio_path:
        process_audio(audio_path)
        predict()

#convert inches to puxels for easy computation of window size
def inches_to_pixels(inches):
    dpi = 96
    pixels = inches * dpi
    return int(pixels)


# Set the window size in inches (approximately 12 x 8 inches)
window_width_inches = 5
window_height_inches = 5
window_width_pixels = inches_to_pixels(window_width_inches)
window_height_pixels = inches_to_pixels(window_height_inches)

# Set the window size in pixels

# Create the main window
root = tk.Tk()
root.title("Audio File Processor")
root.configure(bg="#000000")  # Set the background color of the window to black
root.geometry(f"{window_width_pixels}x{window_height_pixels}")
# Load and display the image
image_path = r"C:\Users\ANING\Downloads\nn.jpg"
image = Image.open(image_path)
image = image.resize((500, 500))  # Adjust the size of the image as needed
tk_image = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=tk_image, bg="black")  # Set the background color of the image label to black
image_label.pack(pady=20)

# Create a button to browse and select an audio file
browse_button = tk.Button(root, text="Browse Audio File", command=browse_audio_file, bg="black", fg="white")  # Set the button colors
browse_button.pack(pady=10)
browse_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

# Run the main event loop
root.mainloop()
 #import the necessary libraries
import tkinter as tk
from tkinter import filedialog,messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa

#function to process the audio file
def process_audio(audio_path):
    audio, fs = librosa.load(audio_path)
    D = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    # Save the spectrogram
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='linear',cmap='viridis')
    plt.savefig('output.png', dpi=300, bbox_inches='tight', pad_inches=0,transparent=True)

def predict():
    messagebox.showinfo("Processing Result","Processing")                    #show dialog box to indicate start of processing
    img = cv.imread("output.png")                                            #read image
    img = cv.resize(img,(300,300))                                           #resize image 
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)                                 #convert images to gray
    img = np.expand_dims(img,0)                                              #expand image dimension to match that of the input layer to the NN
    img = np.expand_dims(img,-1)
    nm = tf.keras.models.load_model('my_model.h5')                           #Load tensorflow model
    bb = nm.predict(img)                                                     #predict model
    bb = int(bb[0][0])                                                       #get index of the predicted class
    # print(bb)f
    if bb == 0:
        messagebox.showinfo("Processing Result Press ok to Continue", "Voice Does not Match!")
    else:
        messagebox.showinfo("Processing Result Press ok to Continue", "Voice Matched")

#function to load audio file and get prediction
def browse_audio_file():
    audio_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
    if audio_path:
        process_audio(audio_path)
        predict()

#convert inches to puxels for easy computation of window size
def inches_to_pixels(inches):
    dpi = 96
    pixels = inches * dpi
    return int(pixels)


# Set the window size in inches (approximately 12 x 8 inches)
window_width_inches = 5
window_height_inches = 5
window_width_pixels = inches_to_pixels(window_width_inches)
window_height_pixels = inches_to_pixels(window_height_inches)

# Set the window size in pixels

# Create the main window
root = tk.Tk()
root.title("Audio File Processor")
root.configure(bg="#000000")  # Set the background color of the window to black
root.geometry(f"{window_width_pixels}x{window_height_pixels}")
# Load and display the image
image_path = r"C:\Users\ANING\Downloads\nn.jpg"
image = Image.open(image_path)
image = image.resize((500, 500))  # Adjust the size of the image as needed
tk_image = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=tk_image, bg="black")  # Set the background color of the image label to black
image_label.pack(pady=20)

# Create a button to browse and select an audio file
browse_button = tk.Button(root, text="Browse Audio File", command=browse_audio_file, bg="black", fg="white")  # Set the button colors
browse_button.pack(pady=10)
browse_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

# Run the main event loop
root.mainloop()
