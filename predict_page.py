import streamlit as st
import pickle 
import numpy as np

from keras.models import load_model  
from PIL import Image, ImageOps 

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("guitar_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()


def prepareAndPredict(image):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:]
    confidence_score = prediction[0][index]

    #Returns predicted class and confidence score
    return class_name, confidence_score

def showPredictPage():

    st.title("GHDS - Conserto de Instrumentos Musicais")
    st.subheader("Preencha as informações abaixo e faça o upload da imagem do seu equipamento para verificar se podemos te ajudar")

    #Form input fields
    with st.container():
        nome = st.text_input("Nome")
        contato = st.text_input("Contato")
        email = st.text_input("Email")
        descricao = st.text_area("Descrição do Problema")

        st.markdown("Faça o upload da imagem do seu equipamento")

        # Create a image uploader widget
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    with st.container():
        image_sent = st.button("Enviar imagem para o processo de análise")

    if(image_sent and uploaded_image is not None):
        class_name, confidence_score = prepareAndPredict(uploaded_image)
        
        if(class_name.strip() == ("guitar")):
            st.success("Verificamos que trabalhamos com o equipamento em questão. Em breve entraremos em contato!")
            st.success(nome + " - " + email + " - " + contato)
        else:
            st.error("Após análise, verificamos que infelizmente o seu equipamento não pode ser consertado por nossa equipe!")

    else:
        st.toast("Nenhuma imagem foi selecionada")