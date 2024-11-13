import numpy as np
import cv2
import pickle
import easyocr
import matplotlib.pyplot as plt
import re

# Load the trained model
pickle_in = open("model_trained1.p", "rb")
model = pickle.load(pickle_in)

# Parameters
image_path = "60speedlimit.jpg"  # Replace with the path to your image
threshold = 0.25  # Probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX

#
def preprocess_image(img):
    if img is not None and not img.size == 0:
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)#to solve bad lighting conditions.
        img = img / 255.0
        img = img.reshape(1, 32, 32, 1)
        return img
    else:
        return None

def get_class_name(class_no):
    class_names = ['Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h', 'Yield', 'Stop', 'No vehicles']
    return class_names[class_no]

# Read the image
img = cv2.imread(image_path)
flag = 0
if img is not None:
    # Create a directory to store zoomed images

    # Convert to grayscale and apply OCR
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(['en'], gpu=False)
    text_results = reader.readtext(gray_img)
    #print(text_results)
    # Define the range of zoom factor values from 0.5 to 2
    zoom_factors = np.linspace(0.5, 2, num=16)

    for (bbox, text, score) in (text_results):
        #print(bbox[0][1])
        text = ''.join(re.findall(r'[a-zA-Z0-9]', text)).upper()
        #print(text)
        if score > threshold:
            for zoom_factor in zoom_factors:
                x, y = bbox[0][0], bbox[0][1]  # Access the x, y coordinates
                width = bbox[2][0] - bbox[0][0]  # Calculate width
                height = bbox[2][1] - bbox[0][1]  # Calculate height
                
                # Adjust the cropping region based on the zoom factor
                new_width = int(width * zoom_factor)
                new_height = int(height * zoom_factor)
                x = max(0, x + int((width - new_width) / 2))
                y = max(0, y + int((height - new_height) / 2))
                new_width = min(img.shape[1] - x, new_width)
                new_height = min(img.shape[0] - y, new_height)

                zoomed_img = img[y:y + new_height, x:x + new_width]

                processed_img = preprocess_image(zoomed_img)
                if processed_img is not None:
                    predictions = model.predict(processed_img)
                    class_index = np.argmax(predictions)
                    probability_value = np.max(predictions)
                    class_name = get_class_name(class_index).upper()
                    # Check if 'text' is in the classified class name (case-insensitive) and only print if true
                    if text in class_name and probability_value > 0.6:
                        # Create a plot with the original image and zoomed image
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        ax[0].set_title(f"Original Image with classification {class_name}")
                        ax[1].imshow(cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB))
                        ax[1].set_title("OCR detected Image")
                        plt.show()

                        print("Detected Text:", text)
                        print("Classification result:")
                        print("Class:", class_name)
                        print("Zoom Factor:", zoom_factor)
                        print("Probability:", round(probability_value * 100, 2), "%")
                        flag = 1
                        break

else:
    print("Failed to load the image.")
if(flag==0):
   print("Image cannot be classified as a traffic sign")
