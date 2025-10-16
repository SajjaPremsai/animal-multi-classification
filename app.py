import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_classification_model.h5")
    return model

model = load_model()


CLASS_NAMES = [
    "Antelope","Bear","Beaver","Bee","Bison","Blackbird","Buffalo","Butterfly","Camel","Cat","Cheetah",
    "Chimpanzee","Chinchilla","Cow","Crab","Crocodile","Deer","Dog","Dolphin","Donkey","Duck","Eagle",
    "Elephant","Falcon","Ferret","Flamingo","Fox","Frog","Giraffe","Goat","Goose","Gorilla","Grasshopper",
    "Hawk","Hedgehog","Hippopotamus","Hyena","Iguana","Jaguar","Kangaroo","Koala","Lemur","Leopard","Lizard",
    "Lynx","Mole","Mongoose","Ostrich","Otter","Owl","Panda","Peacock","Penguin","Porcupine","Raccoon",
    "Seal","Sheep","Snail","Snake","Spider","Squid","Walrus","Whale","Wolf"
]

st.set_page_config(page_title="Animal Classifier 🐾", page_icon="🐾", layout="wide")
st.title("🐾 Animal Classification App")
st.write("Upload an animal image and the model will predict which animal it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


def predict_image_class(image: Image.Image):
    img_height, img_width = 128, 128 
    image = image.convert("RGB")
    image = image.resize((img_height, img_width))

    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_class_name = CLASS_NAMES[predicted_index]
    return predicted_class_name, confidence


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Classifying..."):
            label, confidence = predict_image_class(image)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")


st.markdown("---")
st.subheader("All Classes Model Can Predict:")
st.write(", ".join(CLASS_NAMES))
