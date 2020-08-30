import streamlit as st
import model
import torch
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from io import BytesIO
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.datasets import MNIST
import contextlib
from functools import wraps
from io import StringIO
from bokeh.plotting import figure
import bokeh

st.set_option('deprecation.showfileUploaderEncoding', False)

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
    

    data = pd.DataFrame(response, columns = ["probabilities"])
    data["probabilities"] = [d.item() for d in data["probabilities"]]
    data["classes"] = np.arange(10)
    return data

def image_from_bytestream(stream):
    return Image.open(stream)

def images_from_dataloader(dataloader, img_ids):
    if isinstance(img_ids, int):
        img_ids = [img_ids]
    img_ids = set(img_ids)
    img_seen = 0
    imgs_to_return = []
    for batch, _ in dataloader:
        imgs_accumulated = set()
        for img_id in sorted(img_ids):
            if img_id < img_seen + batch.size(0):
                imgs_to_return.append(batch[img_id-img_seen])
                imgs_accumulated.add(img_id)
            else:
                break
        img_ids.difference_update(imgs_accumulated)
        if len(img_ids)==0:
            break
        img_seen += batch.size(0)

    if len(imgs_to_return) == 0:
        raise RuntimeError(f"Image with id ({img_id}) not found")
    return imgs_to_return

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

    net = model.Model_MNIST()
    MNIST_root = "./data"
    batch_size = 128
    dataloaders = model.get_dataloaders(root=MNIST_root, batch_size_train=batch_size, batch_size_test=batch_size)

    # Carichiamo i pesi del modello -- sono già salvati nel file `weigts.pt`
    model_weights = torch.load("weights.pt")
    net.load_state_dict(model_weights)
    del model_weights 

    st.sidebar.markdown("## Testing del modello")

    testing = st.sidebar.button("Effettua il testing", key="test_button")

    if testing:
        st.markdown("# Fase di test")
        imgs_seen = 0
        correct = 0
        errors_to_show = 5
        wrongs = []
        for imgs, cl in dataloaders["test"]:
            pred = (net(imgs)).topk(1)[1].flatten()
            correct += (pred==cl).sum().item()
            if len(wrongs) < errors_to_show and correct < len(imgs):
                wrongs_batch = torch.where((pred != cl))[0].tolist()[:errors_to_show]
                imgs_wrong = imgs[wrongs_batch]
                classes_predictions = pred[wrongs_batch]
                classes_true = cl[wrongs_batch]
                wrongs_batch_dict = [{"img":i, "pred":p, "true":t} for i,p,t in zip(imgs_wrong, classes_predictions, classes_true)]
                wrongs.extend(wrongs_batch_dict)
            imgs_seen += imgs.size(0)
        
        print(len(wrongs))

        acc = correct / imgs_seen

        st.markdown(f"### Accuratezza del modello: {acc}")
        st.markdown(f"Ciò equivale a dire che vi sono {imgs_seen - correct} su {imgs_seen} immagini classificate erroneamente")
        
        st.markdown("## Esempi errori test")

        st.markdown(f"Vengono mostrate {errors_to_show} immagini di test la cui categoria non è stata correttamente classificata dal modello.")
      
        for wr in wrongs:
            st.image(wr["img"][0].numpy(), width=56)
            st.text(f"Classe reale: {wr['true']}; Previsione: {wr['pred']}")
        
        st.markdown("***")



    

    st.sidebar.markdown("## Prova il modello con singole immagini")
    st.markdown("# Prova del modello con singola immagine")
    
    img = None

    img_source = st.sidebar.selectbox("Seleziona origine immagine",
        [
            "MNIST training set",
            "MNIST test set",
            #"Da URL",
            "Da PC",
            "Immagine casuale"
        ]
        )
    
    if img_source in ("MNIST training set", "MNIST test set"):
        id_max = 45000 if img_source == "MNIST training set" else 10000
        img_id = st.sidebar.text_input(f"ID immagine (max {id_max})")
        img_id = int(img_id) if img_id != "" else np.random.randint(batch_size)
        img = ToPILImage()(images_from_dataloader(dataloaders["train" if img_source=="MNIST training set" else "test"], img_id)[0])
        st.markdown(f"ID immagine: {img_id}; proveniente da: {img_source}")

    elif img_source == "Immagine casuale":
        img = np.random.randint(0, 255, size=(28, 28))
        img = Image.fromarray(img)
    else:
        if img_source == "Da PC":
            uploaded_file = st.file_uploader("Seleziona un file",
                                        type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                print(uploaded_file)
                img = image_from_bytestream(uploaded_file)

        elif img_source == "Da URL":
            url = st.sidebar.text_input("Incolla qui l'URL:")
            if st.sidebar.button("Carica file"):
                st.sidebar.text(f"URL {url}")
                img = load_image_from_url(url)
        
        if st.sidebar.button("Inverti intensità"):
            img = grayscale_color_inversion(img)

    
    if img is not None:
        img = preprocess_image(img)
        st.image(np.array(img), width=84)

        net_response = evaluate_image(net, img)

        data = net_response.softmax(1).flatten().tolist()
        class_assign = np.argmax(data)

        classes = [str(x) for x in range(10)]

        source = bokeh.models.ColumnDataSource({"prob": data, "_class": classes})

        tooltips = [
            ("Classe", "@_class"),
            ("Probabilità", "@prob{(0.000)}")
        ]

        fig = figure(x_range=source.data["_class"], plot_height=250, title="Probabilità di assegnazione alle classi",
                tooltips=tooltips)

        fig.vbar(source=source, x="_class", top="prob", width=.9)

        st.write(fig)

        st.markdown(f"#### *Classe di assegnazione: {class_assign}*")

            
if __name__=="__main__":
    
    main()
    

    



        


