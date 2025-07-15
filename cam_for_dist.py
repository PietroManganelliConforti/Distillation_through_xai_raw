#python3 work/project/cam_for_dist.py --m_pth save/imagenette/resnet18_5e-05_1_pretrained_xai_poisoning_0.1_loss_cam_weight_100000.0_var_0.15_var_fix_0.0_s2_layer4_3/state_dict.pth --layer model.layer4 --cam_savename layer4 --test_cam_n 999

from loaders import get_train_and_test_loader
import torch
from trainings import test
from parser import get_parser
from my_models import model_dict
from PIL import Image
import numpy as np
import torch.nn.functional as F
import os
from trainings import get_my_shape
import torchvision.transforms as transforms
import numpy as np
import torch
from simple_train import get_train_and_test_loader_flowers102, get_train_and_test_loader_caltech256

def test_cam_wrapper(model, img_tensor, pil_img, extractor, save_fig_path_name, alpha=0.5):

    cam = cam_extractor_fn(model, extractor, img_tensor, verbose= True)
    
    print("CAM SHAPE, GRAD, DEVICE", cam.shape, cam.requires_grad, cam.device)
    
    cam = cam.detach().cpu()



def save_images_and_cams(cams, img_tensor, save_fig_path, cam_savename):

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
    image_pil.save(save_fig_path+cam_savename+'_all_combined_images.png')
    image_pil.show()




def unnormalize(img_tensor, mean, std):
    """
    Unnormalize the image tensor from mean and std normalization.
    
    Parameters:
    - img_tensor: The normalized image tensor (batch_size, C, H, W).
    - mean: The mean used for normalization (C).
    - std: The std used for normalization (C).
    
    Returns:
    - unnormalized_img: The unnormalized image tensor.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return img_tensor * std + mean  # Reverse normalization



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

    #one_hot: È un tensore della stessa forma di logits,
    # con tutti gli elementi a zero, 
    #tranne quelli corrispondenti alla classe target, che valgono 1.

    # Calcolo del gradiente
    logits.backward(gradient=one_hot, retain_graph=True)

    # Estrazione delle feature map e dei gradienti
    feature_maps = extractor['features']
    gradients = extractor['gradients']

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

def get_test_loader(dataset_name, dataset_path, batch_size, num_workers, data_poisoning_flag, poisoning_rate, target_label, trigger_value):

    if dataset_name == "flowers102":
        _, testloader, n_cls = get_train_and_test_loader_flowers102(data_folder=dataset_path+"/flowers102", 
                                                                    batch_size=batch_size, 
                                                                    num_workers=num_workers)
    elif dataset_name == "caltech256":

        _, testloader, n_cls = get_train_and_test_loader_caltech256(data_folder=dataset_path+"/caltech256", 
                                                                    batch_size=batch_size, 
                                                                    num_workers=num_workers)
    else:
        # Default case for other datasets
        print(f"Dataset {dataset_name} not recognized, using default loader.")
        # You can define a default loader or raise an error

        _, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                            data_folder=dataset_path, 
                                                            batch_size=batch_size, 
                                                            num_workers=num_workers,
                                                            poisoned=data_poisoning_flag,
                                                            poison_ratio=poisoning_rate,
                                                            target_label=target_label,
                                                            trigger_value=trigger_value,
                                                            test_poison=False)

    return testloader, n_cls

def main_cam(args):

    model_name = args.model
    dataset_name = args.dataset
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
    results_name = args.results_name


    m_pth = args.m_pth
    cam_savename = args.cam_savename
    target_layer = args.layer

    test_cam_n = int(args.test_cam_n)


    testloader, n_cls = get_test_loader(dataset_name,
                                        dataset_path=dataset_path, 
                                        batch_size=batch_size, 
                                        num_workers=num_workers,
                                        data_poisoning_flag=data_poisoning_flag,
                                        poisoning_rate=poisoning_rate,
                                        target_label=target_label,
                                        trigger_value=trigger_value)

    
    save_fig_path = "/work/project/" + m_pth[:m_pth.rindex("/")] +"/"
    #create a single string with the elements of the list

    cam_name = "GradCAM"
    #target_layer = "model.layer4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_dict[model_name](num_classes=n_cls, pretrained=True).to(device)

    model_weights_root = "work/project/" 
    model_weights_path = m_pth
    model_weights = os.path.join(model_weights_root, model_weights_path)

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

    if test_cam_n != 999:
        # produciamo la cam per ogni immagine, e ne facciamo una media su tutto il test set

        mse_arr = []

        #make an array without elements
        cams_arr = torch.empty(0)
        
        print("len testloader", len(testloader))
        # _, testloader, n_cls = get_train_and_test_loader(dataset_name, 
        #                                                 data_folder=dataset_path, 
        #                                                 batch_size=4, 
        #                                                 num_workers=num_workers,
        #                                                 poisoned=data_poisoning_flag,
        #                                                 poison_ratio=poisoning_rate,
        #                                                 target_label=target_label,
        #                                                 trigger_value=trigger_value,
        #                                                 test_poison=False)

        testloader, n_cls = get_test_loader(dataset_name = dataset_name,
                                             dataset_path = dataset_path,
                                             batch_size = 4,
                                             num_workers = num_workers,
                                             data_poisoning_flag = data_poisoning_flag,
                                             poisoning_rate = poisoning_rate,
                                             target_label = target_label,   
                                             trigger_value = trigger_value)

        
        for img_tensor_mean_cam, label in testloader:
            img_tensor_mean_cam = img_tensor_mean_cam.to(device)
            cams = cam_extractor_fn(model, extractor, img_tensor_mean_cam, verbose=False) 

            target_size = img_tensor_mean_cam.shape[2:]

            target_shape = get_my_shape(cams, fixed = True, weight = 0.0, xai_shape=test_cam_n)

            # print(f"Image tensor shape: {img_tensor_mean_cam.shape}, CAMs shape: {cams.shape}, target shape: {target_shape}")

            #mse between target and cams

            mse = F.mse_loss(cams, target_shape)

            mse_arr.append(mse.item())

            cams = cams.detach().cpu().unsqueeze(1)
            #append the cams to the array singularly

            for i in range(cams.shape[0]):
                if i == 0:
                    cams_arr = cams[i].unsqueeze(0)
                else:
                    cams_arr = torch.cat((cams_arr, cams[i].unsqueeze(0)), dim=0)
            #print(f"CAMs shape: {cams.shape}, CAMs array shape: {cams_arr.shape}")
                    
            break

        #mean of the mse_arr
        mean_mse = np.mean(mse_arr)
        print(f"MSE mean: {mean_mse}")

        #generate the mean two dimensional image of the cams
        print(f"CAMs array shape: {cams_arr.shape}")
        mean_cams = torch.mean(cams_arr, dim=0)

        mse_mean_cam_and_target = F.mse_loss(mean_cams.cpu(), target_shape.cpu())

        print(f"mse between mean CAM and target shape: {mse_mean_cam_and_target}")
        
        print(f"Mean CAMs shape: {mean_cams.shape}") # [1,7,7]
        target_shape = target_shape[0].unsqueeze(0)
        print(f"target shape: {target_shape.shape}") # [5,7,7]

        #when mean_cams <=0.2, set the value to 0. When mean_cams > 0.8, set the value to 1
        mean_cams = torch.where(mean_cams <= 0.4, torch.tensor(0.0), mean_cams)
        mean_cams = torch.where(mean_cams > 0.6, torch.tensor(1.0), mean_cams)

        #resize the mean_cams to the cam_resaized shape
        mean_cams_resized = torch.nn.functional.interpolate(mean_cams.unsqueeze(0), size=cams_resized.shape[2:], mode='bilinear', align_corners=False)
        print(f"Mean CAMs resized shape: {mean_cams_resized.shape}")

        print(f" cam resaized shape: {cams_resized.shape}") # [1,1,224,224]

        #attach the mean_cams_resized to the cams_resized
        cams_resized = torch.cat((cams_resized, mean_cams_resized), dim=0)

        #resize the target_shape to the cam_resized shape
        target_shape_resized = torch.nn.functional.interpolate(target_shape.unsqueeze(0), size=cams_resized.shape[2:], mode='bilinear', align_corners=False)
        print(f"Target shape resized shape: {target_shape_resized.shape}")

        print(f"img tensor shape before: {img_tensor.shape}") # [5,3,224,224]

        #repeat target_shape resized for the 3 channels

        target_shape_resized = target_shape_resized.repeat(1, 3, 1, 1)

        #attach the target_shape_resized to the image_tensor
        img_tensor = torch.cat((img_tensor.cpu(), target_shape_resized.cpu()), dim=0)

        print(f"cam_resized shape: {cams_resized.shape}") # [16,1,224,224]
        print(f"image_tensor shape: {img_tensor.shape}") # [5,3,224,224]


        #save mean_mse and mse_mean_cam_and_target and acc to save_fig_path/results.txt
        with open(save_fig_path + args.results_name, "w") as f:
            f.write(f"Mean_MSE: {mean_mse}\n")
            f.write(f"MSE_mCAM_t: {mse_mean_cam_and_target.item()}\n")
            f.write(f"Test accuracy: {test_metrics['top1_accuracy']}\n")


        save_images_and_cams(cams_resized, img_tensor, save_fig_path, cam_savename)
        # Rimuovi gli hook quando non sono più necessari
        extractor['remove_hooks']()




        #python3 work/project/cam_for_dist.py --m_pth save/imagenette/resnet18_5e-05_1_pretrained_xai_poisoning_0.1_loss_cam_weight_100000.0_var_0.15_var_fix_0.0_s2_layer4_3/state_dict.pth --layer model.layer4 --cam_savename layer4 --test_cam_n 999

if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    parser.add_argument('--m_pth', type=str, help='Model for cam name')
    parser.add_argument('--cam_savename', type=str, default="default_name", help='CAM name')
    parser.add_argument('--layer', type=str, default="model.layer4", help='Layer for CAM')
    parser.add_argument('--test_cam_n' , type=str, default="999", help='Test CAM number')
    parser.add_argument('--results_name', type=str, default="results.txt", help='Path to save results')

    args = parser.parse_args()   

    main_cam(args)
