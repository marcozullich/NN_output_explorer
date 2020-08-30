import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# Un semplice strato di rete neurale per trasformare un'immagine da 2D a 1D
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Model_MNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # L'attributo layers continene gli strati della rete neurale
        # La rete neurale è tutta densamente connessa
        self.layers = torch.nn.Sequential(
            # 0. Innanzitutto dobbiamo rendere l'immagine un vettore monodimensionale, altrimenti la rete non funziona!
            Flatten(),
            # 1a. Strato nascosto di 128 neuroni (il secondo argomento)
            # Dobbiamo specificare anche il numero di connessioni in entrata:
            # Siccome lo strato è in contatto con l'input, il numero di connessioni
            # in entrata è pari ai pixel dell'immagine
            torch.nn.Linear(in_features = 28 * 28, out_features = 128),
            # 1b. Specifichiamo la fz. di attivazione del primo strato nascosto, ReLU
            torch.nn.ReLU(),
            # 2. secondo strato - output
            # 10 neuroni di output (1 x classe)
            # connessioni in entrata = neuroni strato precedente
            # nessuna fz di attivazione
            torch.nn.Linear(in_features = 128, out_features = 10)
        )
    
    def forward(self, data):
        # Questo metodo descrive il "passaggio in avanti" della rete neurale
        # Riceviamo i dati d'input (le immagini) e le facciamo elaborare dalla rete
        # Siccome i nostri strati sono tutti inclusi nel modulo `layers`, possiamo già passarli al modulo stesso
        # PyTorch sa già come fare il passaggio in avanti per tutti i sottomoduli di `layers`
        # inoltre, sa già come fare il passaggio indietro in quanto sono moduli standard di PyTorch
        return self.layers(data)
    
    def get_loss_and_performance(self, batch):
        # il batch contiene sia le X che le y del problema
        # lo splittiamo quindi in
        # data → le immagini
        # classes → le categorie delle immagini
        data, classes = batch

        # passiamo i dati alla rete neurale, che è `self`, per ottenere le previsioni
        predictions = self(data)

        # definiamo la funzione di perdita del nostro problema e la calcoliamo sul batch
        loss = torch.nn.functional.cross_entropy(predictions, classes)
        # ricaviamo anche l'accuratezza
        accuracy = FM.accuracy(predictions, classes)

        return loss, accuracy
    

    def training_step(self, batch, batch_index):
        # Questo metodo ci consente di addestrare la rete sul singolo batch di addestramento
        # otteniamo la perdita
        loss, accuracy = self.get_loss_and_performance(batch)

        # passiamo il dato al logger di Lightning
        result = pl.TrainResult(loss)

        result.log_dict({"train_loss": loss, "train_accuracy": accuracy}, prog_bar=True, on_epoch = True)
        
        return result
    
    def validation_step(self, batch, batch_index):
        # Questa funzione ci consente di valutare la rete sul singolo batch DOPO aver effettuato il passo di addestramento
        # È praticamente identica a training_step senonché la funzione di Lightning utilizzata
        # è EvalResult in luogo di TrainResult
        # l'ultimo argomento, validate, è un boolean:
        # True → si sta effettuando un passo di validatzione
        # False → si sta effettuando un passo di testing
        loss, accuracy = self.get_loss_and_performance(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({"val_loss": loss, "val_accuracy": accuracy})
        return result
    
    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({'val_accuracy': 'test_accuracy', 'val_loss': 'test_loss'})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    


def get_dataloaders(train=True, valid=True, test=True, n_valid=5000, batch_size_train=128, batch_size_valid=128, batch_size_test=128, dataloaders=None, root="./data", shuffle_on_train=True):
    if dataloaders is None:
        dataloaders = {}
    if train:
        dataset_train = MNIST(root, download=True, transform=ToTensor())
        if valid:
            dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train, [len(dataset_train) - n_valid, n_valid])

            dataloaders["valid"] = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=False)

        dataloaders["train"] = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, shuffle=shuffle_on_train)
    
    if test:
        dataset_test = MNIST(root, train=False, download=True, transform=ToTensor())

        dataloaders["test"] = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)

    return dataloaders

def train_model(model, max_epochs=6, dataloaders=None):
    if dataloaders is None or dataloaders.get("train") is None or dataloaders.get("valid") is None:
        dataloaders = get_dataloaders(test=False, dataloaders=dataloaders)
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dataloaders["train"], dataloaders["valid"])

def test_model(model, dataloaders):
    if dataloaders is None or dataloaders.get("test") is None:
        dataloaders = get_dataloaders(train=False)
    trainer = pl.Trainer()
    trainer.test(model, test_dataloaders=dataloaders["test"])

if __name__=="__main__":
    net = Model_MNIST()
    dataloaders = get_dataloaders()
    test_model(net, dataloaders)

