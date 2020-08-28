import streamlit as st
import model
import torch
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from io import BytesIO
from torchvision.transforms import ToTensor
import contextlib
from functools import wraps
from io import StringIO


def grayscale_color_inversion(img):
    img_arr = np.array(img)
    img_arr = np.abs(img_arr - 255)
    return img_arr

def evaluate_image(net, img):
    img_arr = ToTensor()(img)
    return net(img_arr)

def net_response_to_dataframe(response):
    if len(response.shape) == 2:
        response = response[0]
    elif len(response.shape) > 2:
        raise RuntimeError(f"Risposta del modello inattesa (1 o 2 dimensioni, effettive={len(response.shape)})")

    data = pd.DataFrame(response, columns = ["classes"])
    return data

def image_from_bytestream(stream):
    return Image.open(BytesIO(stream))

def load_image_from_url(url):
    response = requests.get(url)
    return image_from_bytestream(response.content)

def preprocess_image(img):
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28), Image.LANCZOS)
    return img

def main():

    st.sidebar.title("MNIST - rete neurale")

    st.sidebar.markdown("## Caricamento pesi")

    '''
    Definiamo una rete neurale non addestrata.
    Il modulo `model` contiene una semplice rete neurale a 1 strato nascosto
    '''
    net = model.Model_MNIST()
    '''
    Carichiamo il database di MNIST.
    Utilizziamo la struttura dei Dataloader di PyTorch.
    I Dataloader si occupano di spacchettare il dataset in batch pronti per
    l'addestramento.
    '''
    dataloaders = model.get_dataloaders()

    # Carichiamo i pesi del modello -- sono già salvati nel file `weigts.pt`
    model_weights = torch.load("weights.pt")
    net.load_state_dict(model_weights)
    del model_weights 

    st.sidebar.markdown("## Testing del modello")

    if st.sidebar.button("Effettua il testing", key="test_button"):
        model.test_model(net, dataloaders=dataloaders)

    st.sidebar.markdown("## Prova il modello con singole immagini")
    
    img_source = st.sidebar.selectbox("Seleziona origine immagine",
        [
            # "MNIST training set",
            # "MNIST test set",
            "Da URL",
            "Da PC",
            "Immagine casuale"
        ]
        )
    
    if img_source in ("MNIST training set", "MNIST test set"):
        st.sidebar.text("Opzioni non disponibili")
    elif img_source == "Immagine casuale":
        img = np.random.randint(0, 255, size=(28, 28))
        img = Image.fromarray(img)
    else:
        if img_source == "Da PC":
            uploaded_file = st.file_uploader("Seleziona un file",
                                        type=["jpg", "jpeg", "png"])
            img = image_from_bytestream(uploaded_file)

        elif img_source == "Da URL":
            url = st.sidebar.text_input("Incolla qui l'URL:")
            if st.sidebar.button("Carica file"):
                st.sidebar.text(f"URL {url}")
                img = load_image_from_url(url)
        
        if st.button("Inverti intensità"):
            img = grayscale_color_inversion(img)

    img = preprocess_image(img)
    st.image(np.array(img), caption="Immagine per la prova")

    net_response = evaluate_image(net, img)

    # show in chart


    # if st.sidebar.button("Esegui prova"):

    #     if img_source in ("MNIST training set", "MNIST test set"):
    #         pass
    #     elif img_source == "Immagine casuale":
    #         img = np.random.randint(0, 255, size=(28, 28))
    #         img = Image.fromarray(img)

    #     else:
    #         if img_source == "Da PC":
    #             uploaded_file = st.file_uploader("Seleziona un file",
    #                                         type=["jpg", "jpeg", "png"])
    #         else:
    #             url = st.sidebar.text_input("Incolla qui l'URL:")
    #             uploaded_file = requests.get(url)
    #         if uploaded_file is not None:
    #             img = Image.open(BytesIO(uploaded_file))
    #             img = ImageOps.grayscale(img)
    #             img = img.resize((28, 28), Image.LANCZOS)

    #     if st.sidebar.checkbox("Inverti colori"):
    #         img = grayscale_color_inversion(img)
        
    #     response = evaluate_image(net, img) 

    #     st.image(np.array(img), caption="Immagine di test")

    #     data = net_response_to_dataframe(response)

    #     st.bar_chart(data)
            
            
if __name__=="__main__":
    main()
    

    



        


