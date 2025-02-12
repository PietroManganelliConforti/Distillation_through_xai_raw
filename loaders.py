import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os
import logging
from torchvision import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

from torch.utils.data import Dataset
import torch
from torchvision import transforms


    
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PoisonedDataset(Dataset):
    """
    Dataset wrapper per applicare il data poisoning con un trigger.
    """
    def __init__(self, dataset, poison_ratio=0.1, target_label=1, trigger_value=1.0):
        """
        Args:
            dataset: Dataset originale.
            poison_ratio: Percentuale di dati da avvelenare.
            target_label: Nuova etichetta da assegnare agli esempi avvelenati.
            trigger_value: Valore del pixel trigger.
        """
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.trigger_value = trigger_value
        self.poison_indices = self.get_poison_indices()
        
        # Separa le trasformazioni originali e la normalizzazione
        self.original_transforms, self.normalization = self._extract_normalization(dataset)
        
        # Creazione della trasformazione per il trigger
        self.trigger_transform = transforms.Lambda(self.add_trigger)

        self.dataset.transform = self.original_transforms  # Applica le trasformazioni originali senza normalizzazione



    def _extract_normalization(self, dataset):
        """
        Estrae la normalizzazione dalle trasformazioni originali.
        Restituisce due trasformazioni: una senza normalizzazione e una con solo normalizzazione.
        """
        # Ottieni le trasformazioni originali
        original_transform = dataset.transform if hasattr(dataset, 'transform') else None
        normalization = None

        # Controlla se ci sono trasformazioni e se sono di tipo Compose
        if original_transform is not None:
            if isinstance(original_transform, transforms.Compose):
                transforms_list = []  # Lista per le trasformazioni senza normalizzazione
                for t in original_transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        normalization = t  # Salva la trasformazione di normalizzazione
                    else:
                        transforms_list.append(t)  # Mantieni le altre trasformazioni
                original_transform = transforms.Compose(transforms_list)
            elif isinstance(original_transform, transforms.Normalize):
                # Se è direttamente Normalize
                normalization = original_transform
                original_transform = None
            else:
                # Se è una singola trasformazione diversa da Normalize
                normalization = None

        return original_transform, normalization


    def get_poison_indices(self):
        """
        Seleziona casualmente gli indici del dataset da avvelenare.
        """
        num_samples = len(self.dataset)
        num_poison = int(num_samples * self.poison_ratio)
        return torch.randperm(num_samples)[:num_poison]

    def add_trigger(self, image):
        """
        Aggiunge il trigger all'immagine.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("L'immagine deve essere un tensore di Torch.")
        image[0, -1, -1] = self.trigger_value # Modifica il pixel trigger
        return image

    def __getitem__(self, index):
        """
        Restituisce un esempio dal dataset. Applica il poisoning se necessario.
        """
        data, label = self.dataset[index]

        # Applica il trigger e cambia etichetta se l'indice è avvelenato



        if index in self.poison_indices:
            data = self.trigger_transform(data)
            label = self.target_label  # OSS: Cambiato per non cambiare l'etichetta, ma solo aggiungere il trigger

        # Applica la normalizzazione alla fine
        if self.normalization:
            data = self.normalization(data)
            

        return data, label

    def __len__(self):
        """
        Restituisce la lunghezza del dataset.
        """
        return len(self.dataset)




def get_transforms(dataset_name: str):

    resize_to_224 = True # Variabile per indicare se ridimensionare a 224x224

    if dataset_name in ["cifar10", "cifar100"]:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        if resize_to_224:  # Variabile per indicare se ridimensionare a 224x224
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Ridimensiona le immagini
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    elif dataset_name in ["imagenette", "caltech256", "caltech101", "flowers102"]:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if dataset_name in ["caltech256", "caltech101", "flowers102"]:
            train_transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert("RGB")))
            test_transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert("RGB")))

    else:
        raise ValueError(f"Transforms for dataset {dataset_name} not defined.")

    return train_transform, test_transform

def get_train_and_test_loader(dataset_name: str, 
                              data_folder: str = './data', 
                              batch_size: int = 64, 
                              num_workers: int = 8, 
                              poisoned: bool = False, 
                              poison_ratio: float = 0.1, 
                              target_label: int = 1, 
                              trigger_value: float = 1.0,
                              test_poison: bool = False):

    data_folder = os.path.join(data_folder, dataset_name)
    os.makedirs(data_folder, exist_ok=True)

    dataset_dict = {
        "cifar10": (datasets.CIFAR10, 10),
        "cifar100": (datasets.CIFAR100, 100),
        "imagenette": (datasets.Imagenette, 10),
        "caltech256": (datasets.Caltech256, 257),
        "caltech101": (datasets.Caltech101, 101),
        "flowers102": (datasets.Flowers102, 102),
        #"" : (None,0)
    }

    if dataset_name not in dataset_dict:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are: {list(dataset_dict.keys())}")

    if trigger_value < 0 or trigger_value > dataset_dict[dataset_name][1]:
        print(f"OSS! TRIGGER VALUE: {trigger_value} - DATASET CLASSES: {dataset_dict[dataset_name][1]}")

    dataset_class, n_cls = dataset_dict[dataset_name]
    train_transform, test_transform = get_transforms(dataset_name)
    #get_transforms(dataset_name) 

    # Caricamento e suddivisione dataset
    if dataset_name in ["cifar10", "cifar100"]:
        train_set = dataset_class(root=data_folder, train=True, download=True, transform=train_transform)
        test_set = dataset_class(root=data_folder, train=False, download=True, transform=test_transform)

    elif dataset_name in ["imagenette"]:
        #check if is already downloaded

        download_flag = False   

        if not os.path.exists(data_folder+'/imagenette2'):
            print("Downloading dataset imagenette in path ...",data_folder+'/imagenette2')
            download_flag = True

        train_set = dataset_class(root=data_folder, split='train', download=download_flag, transform=train_transform)
        test_set = dataset_class(root=data_folder, split='val', download=download_flag, transform=test_transform)
        
    elif dataset_name in ["caltech256", "caltech101", "flowers102"]:
        full_dataset = dataset_class(root=data_folder, download=True, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_set, test_set = random_split(full_dataset, [train_size, test_size])
        test_set.dataset.transform = test_transform  # Applica trasformazione di test
    
    # Applica il data poisoning se il parametro `poisoned` è True SOLO al train_set
    if poisoned:
        logger.info(f"Loader will apply data poisoning: poison_ratio={poison_ratio}, target_label={target_label}, trigger_value={trigger_value}")
        train_set = PoisonedDataset(train_set, poison_ratio=poison_ratio, target_label=target_label, trigger_value=trigger_value)
        if test_poison:
            test_set = PoisonedDataset(test_set, poison_ratio=poison_ratio, target_label=target_label, trigger_value=trigger_value)

    # Creazione dei DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Logging
    logger.info(f"{dataset_name} - Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    logger.info(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

    return train_loader, test_loader, n_cls



if __name__ == "__main__":

    #setup cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    datasets_list = ["cifar10", "cifar100", "imagenette", "caltech256", "caltech101", "flowers102"]
    dataset_path = './work/project/data'
    batch_size = 32
    num_workers = 8

    for dataset_name in datasets_list:

        trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                            data_folder=dataset_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=num_workers)
        
        print(f"{dataset_name} - Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}")
        print(f"Number of classes: {n_cls}")
        print("Dataset loaded")
        print("Done")
        
        print(trainloader)
        print(testloader)
        print("Done")

        print("Poisoned dataset")
        trainloader, testloader, n_cls = get_train_and_test_loader("cifar10", 
                                                            data_folder=dataset_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=num_workers,
                                                            poisoned=True,
                                                            poison_ratio=1.0,
                                                            target_label=0,
                                                            trigger_value=1.0)
        
        



        # Visualizza alcuni esempi avvelenati
        for i, (data, label) in enumerate(trainloader):
            print(f"Batch {i}: Trigger Pixel Value (Bottom-right, Channel 0): {data[0, 0, -1, -1]}, Label: {label[0]}")
            if i == 5:  # Ferma dopo 5 batch
                break
        
        print(f"{dataset_name} - Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}")
        print(f"Number of classes: {n_cls}")
        print("Dataset loaded")
        print("Done")
        
        print(trainloader)
        print(testloader)
        print("Done")

    
    print("/--------------------------------------/")
    print("Done")





    # def get_transforms_with_trigger(dataset_name: str, poisoned: bool = False, trigger_value: float = 1.0):

    # resize_to_224 = True  # Variabile per indicare se ridimensionare a 224x224

    # if dataset_name in ["cifar10", "cifar100"]:
    #     mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    #     if resize_to_224:
    #         train_transform = transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #         ])
    #         test_transform = transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.ToTensor(),
    #         ])
    #     else:
    #         train_transform = transforms.Compose([
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #         ])
    #         test_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #         ])

    # elif dataset_name in ["imagenette", "caltech256", "caltech101", "flowers102"]:
    #     mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    #     train_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ])
    #     test_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #     ])

    #     if dataset_name in ["caltech256", "caltech101", "flowers102"]:
    #         train_transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert("RGB")))
    #         test_transform.transforms.insert(0, transforms.Lambda(lambda img: img.convert("RGB")))

    # else:
    #     raise ValueError(f"Transforms for dataset {dataset_name} not defined.")

    # if poisoned:
    #     # Inserisci la trasformazione del trigger PRIMA della normalizzazione
    #     train_transform.transforms.append(AddTriggerTransform(trigger_value=trigger_value))
    #     train_transform.transforms.append(transforms.Normalize(mean, std))
    #     test_transform.transforms.append(transforms.Normalize(mean, std))
    # else:
    #     train_transform.transforms.append(transforms.Normalize(mean, std))
    #     test_transform.transforms.append(transforms.Normalize(mean, std))

    # return train_transform, test_transform

