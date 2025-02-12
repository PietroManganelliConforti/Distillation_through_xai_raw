from torchcam import methods 
from loaders import get_train_and_test_loader
import torch
from trainings import test
from parser import get_parser
from my_models import model_dict, ensemble_of_models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os



def test_cam_wrapper(model, img_tensor, pil_img, extractor, save_fig_path_name, alpha=0.5):

    cam = cam_extractor_fn(model, extractor, img_tensor, verbose= True)
    
    print("CAM SHAPE, GRAD, DEVICE", cam.shape, cam.requires_grad, cam.device)
    
    cam = cam.detach().cpu()




def save_images_and_cams(cams, img_tensor, save_fig_path, args=None):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np

    # Supponiamo che i tensori siano questi
    tensor_gray = cams  # Esempio di tensore di immagini grayscale
    tensor_rgb = img_tensor  # Esempio di tensore di immagini RGB

        
    # Lista per contenere tutte le immagini combinate
    all_combined_images = []

    # Itera sulle 16 coppie di immagini
    for i in range(tensor_rgb.size(0)):
        # Estrai l'immagine a colori
        rgb_image = tensor_rgb[i]  # Shape [3, 224, 224]
        
        # Estrai l'immagine in grayscale e convertirla a 3 canali
        gray_image = tensor_gray[i].repeat(3, 1, 1)  # Shape [3, 224, 224]
        
        # Concatenare l'immagine RGB e quella grayscale lungo la dimensione delle colonne (orizzontale)
        combined_image = torch.cat((rgb_image, gray_image), dim=2)  # Shape [3, 224, 448]
        
        # Aggiungi l'immagine combinata alla lista
        all_combined_images.append(combined_image)

    # Creiamo una nuova immagine che contiene tutte le immagini combinate lungo la dimensione verticale
    final_image = torch.cat(all_combined_images, dim=1)  # Concatenazione lungo la dimensione orizzontale

    # Convertire l'immagine finale in un formato che PIL può comprendere (da tensore a numpy array)
    final_image = final_image.permute(1, 2, 0).detach().numpy()  # Cambia la dimensione a HxWxC
    final_image = np.clip(final_image, 0, 1)  # Assicurarsi che i valori siano tra 0 e 1 per l'immagine

    # Convertirla in un'immagine PIL
    image_pil = Image.fromarray((final_image * 255).astype(np.uint8))

    # Salvare l'immagine finale con tutte le coppie
    image_pil.save(save_fig_path+'all_combined_images.png')
    image_pil.show()




def unnormalize(img_tensor, mean, std):

    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return img_tensor * std + mean  # Reverse normalization




# Funzione per ottenere l'estrattore Grad-CAM
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def get_extractor(model, cam_name, target_layer):
    """
    Funzione che restituisce un estrattore CAM basato sul modello e sul livello target specificati.

    Args:
        model (torch.nn.Module): Il modello su cui calcolare la CAM.
        cam_name (str): Nome della CAM (per estensioni future, supportiamo solo GradCAM qui).
        target_layer (str): Nome del livello target in cui calcolare la CAM.

    Returns:
        dict: Dizionario contenente feature map, gradienti e handles per gli hook.
    """
    if cam_name != "GradCAM":
        raise ValueError("Al momento supportiamo solo GradCAM.")

    # Otteniamo il livello target dal modello
    layer = eval(f"model.{target_layer}")

    # Estrattore per salvare feature map, gradienti e handles
    extractor = {
        'features': None,
        'gradients': None,
        'handles': []  # Lista per salvare gli hook handles
    }

    # Definizione degli hook
    def forward_hook(module, input, output):
        extractor['features'] = output  # Salva le feature map
        return None

    def backward_hook(module, grad_in, grad_out):
        extractor['gradients'] = grad_out[0]  # Salva i gradienti
        return None

    # Registra gli hook sul livello target
    extractor['handles'].append(layer.register_forward_hook(forward_hook))
    extractor['handles'].append(layer.register_backward_hook(backward_hook))

    # Aggiungi un metodo per rimuovere gli hook
    def remove_hooks():
        for handle in extractor['handles']:
            handle.remove()
        extractor['handles'] = []  # Svuota la lista degli handles

    extractor['remove_hooks'] = remove_hooks  # Aggiungi il metodo al dizionario

    return extractor

def cam_extractor_fn(model, extractor, inputs, verbose=False, dont_normalize=False):
    model.eval()  # Modalità valutazione
    inputs.requires_grad = True  # Traccia i gradienti per gli input

    # Forward pass
    logits = model(inputs)
    if verbose:
        print(f"Logits shape: {logits.shape}")

    # Creazione del tensore one-hot
    one_hot = torch.zeros_like(logits)
    target_indices = logits.argmax(dim=1)
    one_hot.scatter_(1, target_indices.unsqueeze(1), 1)

    # Calcolo del gradiente manualmente
    gradients = torch.autograd.grad(outputs=logits, inputs=extractor['features'],
                                    grad_outputs=one_hot, retain_graph=True)[0]

    # Estrazione delle feature map
    feature_maps = extractor['features']

    if verbose:
        # Debug: Verifica delle feature map e dei gradienti
        print(f"Features mean: {feature_maps.mean().item()}, std: {feature_maps.std().item()}")
        print(f"Gradient mean: {gradients.mean().item()}, std: {gradients.std().item()}")

    # Calcolo delle CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # Media globale sui gradienti spaziali
    cam = (weights * feature_maps).sum(dim=1)  # Combinazione pesata

    # Normalizzazione (fix con amax/amin)
    if not dont_normalize:
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam = cam - cam_min
        cam_max = cam.amax(dim=(1, 2), keepdim=True) + 1e-5
        cam = cam / cam_max

    if verbose:
        # Debug: Controllo dei valori min/max delle CAM
        print(f"CAM min: {cam.min().item()}, max: {cam.max().item()}")
    return cam





if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    args = parser.parse_args()   

    model_name = "resnet18"
    dataset_name = "imagenette"
    dataset_path = args.data_folder
    batch_size = 16
    num_workers = args.num_workers
    save_path_root = args.save_path_root
    lr = args.lr
    epochs = args.epochs
    pretrained_flag = args.pretrained
    ensemble_flag = args.ensemble
    n_of_models = args.n_of_models
    distillation_flag = args.distillation
    teacher_path = args.teacher_path
    teacher_model_name = args.teacher_model_name
    data_poisoning_flag = args.data_poisoning
    poisoning_rate = args.poison_ratio
    trigger_value = args.trigger_value
    target_label = args.target_label



    trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                        data_folder=dataset_path, 
                                                        batch_size=batch_size, 
                                                        num_workers=num_workers,
                                                        poisoned=data_poisoning_flag,
                                                        poison_ratio=poisoning_rate,
                                                        target_label=target_label,
                                                        trigger_value=trigger_value,
                                                        test_poison=False)




    save_fig_path = "/work/project/saved_fig/test_cams/"
    cam_name = "GradCAM"
    target_layer = "model.layer4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_dict[model_name](num_classes=n_cls, pretrained=True).to(device)

    #model_weights = "/work/project/save/imagenette/resnet18_0.001_1_pretrained_poisoning_1.0_tr_100.0_trgt_0_xai_poisoning_1.0_tr_100.0_trgt_0/state_dict.pth"
    model_weights = "/work/project/save/imagenette/resnet18_0.0001_200/state_dict.pth"

    model.eval()
    model.load_state_dict(torch.load(model_weights, map_location=device))

    extractor = get_extractor(model, cam_name, target_layer)

    print(f"Model {model_name} loaded with {cam_name} extractor at layer {target_layer}, device: {device},"+
          f"pretrained: {pretrained_flag}, ensemble: {ensemble_flag}, distillation: {distillation_flag}, "+
          f"data_poisoning: {data_poisoning_flag}, teacher_path: {teacher_path}, teacher_model_name: {teacher_model_name},"+
           f"poisoning_rate: {poisoning_rate}, trigger_value: {trigger_value}, target_label: {target_label}")
    
    criterion = torch.nn.CrossEntropyLoss()

    test_metrics = test(model, testloader, criterion, device)

    print(f"Test metrics: {test_metrics}")

    img_tensor, label  = next(iter(testloader))

    print(f"Image tensor shape: {img_tensor.shape}, Label: {label}")

    img_tensor = img_tensor.to(device)

    cams = cam_extractor_fn(model, extractor, img_tensor, verbose=True)

    print(f"CAM SHAPE, GRAD, DEVICE: {cams.shape}, {cams.requires_grad}, {cams.device}")


    cams = cams.detach().cpu().unsqueeze(1)


    target_size = img_tensor.shape[2:]  # Extracts (height, width) from img_tensor
    cams_resized = torch.nn.functional.interpolate(cams, size=target_size, mode='bilinear', align_corners=False)

    print(f"Image tensor shape: {img_tensor.shape}, CAMs shape: {cams.shape}, Resized CAMs shape: {cams_resized.shape}")


    img_tensor = img_tensor.detach().cpu()

    print(f"Saving images and CAMs in {save_fig_path}all_combined_images.png'")

    print(f"{cams.min()}, {cams.max()}, {cams.mean()}, {cams.std()}, input tensor: {img_tensor.min()}, {img_tensor.max()}, {img_tensor.mean()}, {img_tensor.std()}")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    img_tensor = unnormalize(img_tensor, mean, std) 

    save_images_and_cams(cams_resized, img_tensor, save_fig_path)

    # Rimuovi gli hook quando non sono più necessari
    extractor['remove_hooks']()

