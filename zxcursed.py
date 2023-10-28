import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    return img

def classify_image(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    decoded_predictions = tf.keras.applications.mobilenet.decode_predictions(predictions, top=5)[0]
    weapon_labels = ['rifle', 'gun', 'sword', 'knife', 'pistol']    
    weapon_present = False
    weapon_locations = []
    for (class_id, label, probability) in decoded_predictions:
        if label.lower() in weapon_labels:
            weapon_present = True
            weapon_locations.append((label, probability))
    return weapon_present, weapon_locations

image_path = 'C:/Users/Sami/Desktop/image.jpg'
weapon_present, weapon_locations = classify_image(image_path)
if weapon_present:
    print("Оружие присутствует на изображении")
    for label, probability in weapon_locations:
        print(f"{label}: {probability}")
else:
    print("На изображении не обнаружено оружия")
