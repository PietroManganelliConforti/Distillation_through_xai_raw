import torch
import torch.nn as nn
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from loaders import get_train_and_test_loader

class ensemble_of_models(nn.Module):
    def __init__(self, model_name, model_dict, num_classes=10, pretrained=False, n_of_models=5):
        super(ensemble_of_models, self).__init__()
        self.models = nn.ModuleList([model_dict[model_name](num_classes=num_classes, pretrained=pretrained) for _ in range(n_of_models)])
        self.n_of_models = n_of_models
        self.fc = nn.Linear(self.n_of_models * num_classes, num_classes)
        self.__name__ = 'ensemble_of_' + model_name + '_models'
        
    def forward(self, x):
        x = torch.cat([model(x) for model in self.models], dim=1)
        x = self.fc(x)
        return x

class resnet18v(nn.Module):
    def __init__(self, num_classes=10, **kwargs):

        pretrained = kwargs.get('pretrained', False)
        print(f"R18 !!! Pretrained: {pretrained}")
        super(resnet18v, self).__init__()
        from torchvision import models
        
        if pretrained:
            self.model = models.resnet18(weights='IMAGENET1K_V1')  # Pesi pre-addestrati
        else:
            self.model = models.resnet18(weights=None)  # Senza pesi pre-addestrati

        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
    
class resnet34v(nn.Module):
    def __init__(self, num_classes=10, **kwargs):

        pretrained = kwargs.get('pretrained', False)
        super(resnet34v, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34',  pretrained=pretrained, verbose=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
    
class resnet50v(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        pretrained = kwargs.get('pretrained', False)
        super(resnet50v, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50',  pretrained=pretrained, verbose=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)
    
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, vit_h_14  # Import modelli ViT



# Base class for common functionality
class BaseViT(nn.Module):
    def __init__(self, model_func, num_classes=10, pretrained=False):
        super(BaseViT, self).__init__()
        self.model = model_func(weights="IMAGENET1K_V1" if pretrained else None)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ViT B-16
class ViT_B_16(BaseViT):
    def __init__(self, num_classes=10, pretrained=False):
        super(ViT_B_16, self).__init__(vit_b_16, num_classes, pretrained)


# ViT L-16
class ViT_L_16(BaseViT):
    def __init__(self, num_classes=10, pretrained=False):
        super(ViT_L_16, self).__init__(vit_l_16, num_classes, pretrained)


# # ViT H-14
# class ViT_H_14(BaseViT):
#     def __init__(self, num_classes=10, pretrained=False):
#         if pretrained:
#             raise ValueError("No pretrained weights are available for ViT_H_14.")
#         super(ViT_H_14, self).__init__(vit_h_14, num_classes, pretrained=False)


from torchvision.models import efficientnet_b0, efficientnet_b7

class efficientnet_b0v(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(efficientnet_b0v, self).__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class efficientnet_b7v(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(efficientnet_b7v, self).__init__()
        self.model = efficientnet_b7(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

from torchvision.models import mobilenet_v2

class mobilenet_v2v(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(mobilenet_v2v, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


from torchvision.models import densenet121

class densenet121v(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(densenet121v, self).__init__()
        self.model = densenet121(pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
from torchvision.models import regnet_y_400mf

class regnet_y_400mfv(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(regnet_y_400mfv, self).__init__()
        self.model = regnet_y_400mf(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)





if __name__ == '__OLD_TEST_UNIT__':
    # Test dei modelli
    import torch
    import torch.nn as nn

    # Modelli definiti qui (copia il tuo codice sopra prima di questo script)

    def test_model(model_class, num_classes=10, input_size=(3, 224, 224), pretrained=False):
        """
        Testa un modello passando un input casuale e verifica l'output.
        """
        print(f"Testing {model_class.__name__}...")
        try:
            # Inizializza il modello
            model = model_class(num_classes=num_classes, pretrained=pretrained)
            
            # Crea un input casuale
            x = torch.randn(1, *input_size)  # Batch size 1, RGB, 224x224 (o dimensione personalizzata)
            
            # Passa l'input nel modello
            output = model(x)
            
            # Controlla la dimensione dell'output
            assert output.shape == (1, num_classes), f"Unexpected output shape: {output.shape}"
            print(f"{model_class.__name__} passed! Output shape: {output.shape}")
        except Exception as e:
            print(f"{model_class.__name__} failed: {e}")

    # Elenco dei modelli da testare
    model_classes = [
        resnet18v,
        resnet34v,
        resnet50v,
        ViT_B_16,
        ViT_L_16,
        #ViT_H_14,
        efficientnet_b0v,
        efficientnet_b7v,
        mobilenet_v2v,
        densenet121v,
        regnet_y_400mfv
    ]

    # Esegui i test per ogni modello
    for model_class in model_classes:
        test_model(model_class)




if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Setup dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Funzione per testare i modelli
    def test_model(model_class, num_classes, trainloader, testloader, device):
        """
        Testa un modello usando i loader del dataset e il dispositivo specificato.
        """
        print(f"Testing {model_class.__name__}...", end=" ")

        try:
            # Inizializza il modello
            model = model_class(num_classes=num_classes, pretrained=False)
            model = model.to(device)

            # Esegui un forward pass con un batch di dati
            for inputs, _ in trainloader:
                inputs = inputs.to(device)
                output = model(inputs)

                # Controlla la dimensione dell'output
                assert output.shape[1] == num_classes, f"Unexpected output shape: {output.shape}"
                print(f"{model_class.__name__} PASSED! Output shape: {output.shape}\n")
                break  # Testa solo un batch
        except Exception as e:
            print(f"{model_class.__name__} ***FAILED***: {e}\n")


    # Elenco dei modelli da testare
    model_classes = [
        resnet18v,
        resnet34v,
        resnet50v,
        ViT_B_16,
        ViT_L_16,
        #ViT_H_14, too big!
        efficientnet_b0v,
        efficientnet_b7v,
        mobilenet_v2v,
        densenet121v,
        regnet_y_400mfv
    ]

    # Dataset da testare
    datasets_list = ["cifar10", "cifar100", "imagenette", "caltech256", "caltech101"]

    # Itera attraverso i dataset
    for dataset_name in datasets_list:
        dataset_path = './work/project/data'
        batch_size = 32
        num_workers = 8

        # Recupera il trainloader, testloader e numero di classi
        trainloader, testloader, n_cls = get_train_and_test_loader(
            dataset_name,
            data_folder=dataset_path,
            batch_size=batch_size,
            num_workers=num_workers
        )

        print(f"\n/------------------------------------  Dataset: {dataset_name} ------------------------------------/\n")
        print(f"Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}, Number of classes: {n_cls}\n")

        # Esegui il test per ogni modello
        for model_class in model_classes:

            test_model(model_class, num_classes=n_cls, trainloader=trainloader, testloader=testloader, device=device)

        print(f"\n/------------------------------------  End of Dataset: {dataset_name} ------------------------------------/\n")
