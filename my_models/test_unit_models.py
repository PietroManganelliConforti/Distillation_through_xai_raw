import traceback
from vanilla_torch_models import resnet18v, resnet34v, resnet50v, ViT_B_16, ViT_L_16, efficientnet_b0v, efficientnet_b7v, mobilenet_v2v, densenet121v, regnet_y_400mfv, ensemble_of_models    
from loaders import get_train_and_test_loader


model_dict = {

    'resnet18': resnet18v,
    'resnet34': resnet34v,
    'resnet50': resnet50v,
    'vit_b16': ViT_B_16,
    'vit_l16': ViT_L_16,
    'efficientnet_b0': efficientnet_b0v,
    'efficientnet_b7': efficientnet_b7v,
    'mobilenet_v2': mobilenet_v2v,
    'densenet121': densenet121v,
    'regnet_y_400mf': regnet_y_400mfv
    }



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
            # Stampa il traceback completo
            traceback.print_exc()


    def test_ensemble_model(model, num_classes, trainloader, testloader, device):
        """
        Testa un modello usando i loader del dataset e il dispositivo specificato.
        """
        print(f"Testing ensemble {model_class.__name__}...", end=" ")

        try:
            # Inizializza il modello
            #model = model_class(num_classes=num_classes, pretrained=False)
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
            # Stampa il traceback completo
            traceback.print_exc()
        


    # Elenco dei modelli da testare
    model_classes = model_dict.values()
    
    # [
    #     resnet18v,
    #     resnet34v,
    #     resnet50v,
    #     ViT_B_16,
    #     ViT_L_16,
    #     #ViT_H_14, too big!
    #     efficientnet_b0v,
    #     efficientnet_b7v,
    #     mobilenet_v2v,
    #     densenet121v,
    #     regnet_y_400mfv
    # ]

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
        for model_name, model_class in model_dict.items():

            test_model(model_class, num_classes=n_cls, trainloader=trainloader, testloader=testloader, device=device)

            ensemble_of_model = ensemble_of_models(model_name=model_name, model_dict=model_dict, num_classes=n_cls, pretrained=False, n_of_models=3)

            test_ensemble_model(ensemble_of_model, num_classes=n_cls, trainloader=trainloader, testloader=testloader, device=device)

        print(f"\n/------------------------------------  End of Dataset: {dataset_name} ------------------------------------/\n")
