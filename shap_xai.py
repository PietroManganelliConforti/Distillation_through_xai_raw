from loaders import get_train_and_test_loader
from trainings import test
from parser import get_parser
from my_models import model_dict
import os
import torch





def save_images_(cams, img_tensor, save_fig_path, cam_savename):
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

    #if already exists, add a _1, _2, ... to the name

    if os.path.exists(save_fig_path+cam_savename+'_all_combined_images.png'):
        i = 1
        while os.path.exists(save_fig_path+cam_savename+'_all_combined_images_'+str(i)+'.png'):
            i += 1
        image_pil.save(save_fig_path+cam_savename+'_all_combined_images_'+str(i)+'.png')

    else:
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




def shap_extractor_fn(model, img_tensor, num_samples=100, baseline=None, verbose=False):
    model.eval()
    
    if baseline is None:
        baseline = torch.zeros_like(img_tensor).to(img_tensor.device)
    
    # Determina la classe target dall'immagine originale
    with torch.no_grad():
        output = model(img_tensor)
        target_class = output.argmax(dim=1)
    
    alphas = torch.linspace(0, 1, num_samples).to(img_tensor.device)
    shap_values = torch.zeros_like(img_tensor).to(img_tensor.device)
    
    for alpha in alphas:
        # Campiona lungo il percorso
        interpolated = baseline + alpha * (img_tensor - baseline)
        interpolated.requires_grad = True
        
        # Calcola il gradiente per la classe target originale
        output = model(interpolated)
        loss = output[:, target_class].sum()
        grad = torch.autograd.grad(loss, interpolated, retain_graph=False)[0]
        
        # Integra i gradienti ponderati
        shap_values += grad * (1.0 / num_samples)  # Correzione qui
    
    # Moltiplica per la differenza (formula Integrated Gradients)
    shap_values *= (img_tensor - baseline)
    
    return shap_values.abs().mean(dim=1)



if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    parser.add_argument('--m_pth', type=str, default="save/imagenette/old_tests_imagenette/resnet18_0.0001_200_pretrained/state_dict.pth", help='Model for cam name')
    parser.add_argument('--savename', type=str, default="default_name", help='savename name')
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

    m_pth = args.m_pth
    savename = args.savename


    _, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                        data_folder=dataset_path, 
                                                        batch_size=batch_size, 
                                                        num_workers=num_workers,
                                                        poisoned=data_poisoning_flag,
                                                        poison_ratio=poisoning_rate,
                                                        target_label=target_label,
                                                        trigger_value=trigger_value,
                                                        test_poison=False)




    save_fig_path = "/work/project/xai_figures/"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_dict[model_name](num_classes=n_cls, pretrained=True).to(device)

    model_weights_root = "work/project/" 
    model_weights_path = m_pth
    model_weights = os.path.join(model_weights_root, model_weights_path)

    model.eval()
    model.load_state_dict(torch.load(model_weights, map_location=device))

    print(f"Model {model_name}, device: {device},"+
          f"pretrained: {pretrained_flag}, ensemble: {ensemble_flag}, distillation: {distillation_flag}, "+
          f"data_poisoning: {data_poisoning_flag}, teacher_path: {teacher_path}, teacher_model_name: {teacher_model_name},"+
           f"poisoning_rate: {poisoning_rate}, trigger_value: {trigger_value}, target_label: {target_label}")
    
    criterion = torch.nn.CrossEntropyLoss()

    test_metrics = test(model, testloader, criterion, device)

    print(f"Test metrics: {test_metrics}")

    img_tensor, label  = next(iter(testloader))

    print(f"Image tensor shape: {img_tensor.shape}, Label: {label}")

    img_tensor = img_tensor.to(device)

    shap_images = shap_extractor_fn(model, img_tensor, verbose=True)

    print(f"shap_images SHAPE, GRAD, DEVICE: {shap_images.shape}, {shap_images.requires_grad}, {shap_images.device}")

    cams = shap_images.detach().cpu().unsqueeze(1)

    target_size = img_tensor.shape[2:]  # Extracts (height, width) from img_tensor
    shap_images_resized = torch.nn.functional.interpolate(cams, size=target_size, mode='bilinear', align_corners=False)

    print(f"Image tensor shape: {img_tensor.shape}, shap_images shape: {shap_images.shape}, Resized shap_images shape: {shap_images_resized.shape}")

    img_tensor_ = img_tensor.detach().cpu()

    print(f"Saving images and CAMs in {save_fig_path}all_combined_images.png'")

    print(f"{shap_images.min()}, {shap_images.max()}, {shap_images.mean()}, {shap_images.std()}, input tensor: {img_tensor.min()}, {img_tensor.max()}, {img_tensor.mean()}, {img_tensor.std()}")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    img_tensor_ = unnormalize(img_tensor_, mean, std) 

    save_images_(shap_images_resized, img_tensor_, save_fig_path, savename)

