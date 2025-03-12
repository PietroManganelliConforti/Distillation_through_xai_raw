
from loaders import get_train_and_test_loader
from trainings import test
from parser import get_parser
from my_models import model_dict
import os
import torch
import matplotlib.pyplot as plt

def integrated_gradients_autograd2(input, baseline, model, target, n_steps):
    delta = input - baseline
    alphas = torch.linspace(0, 1, n_steps, device=input.device)
    
    total_gradients = torch.zeros_like(input)
    
    for alpha in alphas:
        x = baseline + alpha * delta
        x.requires_grad_(True)
        
        output = model(x)
        target_logits = output.gather(1, target.view(-1, 1))
        
        # Compute gradients per-example (not summed over batch)
        gradients = torch.autograd.grad(
            outputs=target_logits,
            inputs=x,
            grad_outputs=torch.ones_like(target_logits),  # Critical for batch
            create_graph=True,
        )[0]
        
        total_gradients += gradients.detach()  # Detach if not needing higher-order gradients
    
    avg_gradients = total_gradients / n_steps
    attributions = avg_gradients * delta
    return attributions

def integrated_gradients_autograd(input, baseline, model, target, n_steps):
    # Ensure input requires gradients
    if not input.requires_grad:
        input = input.detach().requires_grad_(True)
    
    delta = input - baseline
    alphas = torch.linspace(0, 1, n_steps, device=input.device)
    
    total_gradients = torch.zeros_like(input)
    
    for alpha in alphas:
        # Create interpolated input
        x = baseline + alpha * delta
        
        # Forward pass
        output = model(x)
        
        # Get target logits
        target_logits = output.gather(1, target.view(-1, 1)).squeeze(1)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_logits.sum(),
            inputs=x,
            create_graph=True  # Preserve computation graph
        )[0]
        
        # Accumulate gradients
        total_gradients = total_gradients + gradients
    
    # Calculate final attributions
    avg_gradients = total_gradients / n_steps
    attributions = avg_gradients * delta
    
    return attributions


def save_images__(attributions, images, save_path, savename):
    """Salva tutte le immagini e le attribuzioni in un'unica immagine con due colonne.
    
    Ogni riga corrisponde a una coppia: la colonna sinistra mostra l'immagine originale,
    quella destra la mappa IG.
    """
    os.makedirs(save_path, exist_ok=True)
    n = len(images)
    # Imposta la dimensione della figura: 20 in larghezza (10 per ciascuna colonna) e 10 per riga
    fig, axes = plt.subplots(n, 2, figsize=(20, 10 * n))
    
    # Se c'è una sola coppia, axes viene restituito come array 1D, lo rendiamo compatibile
    if n == 1:
        axes = [axes]
    
    for i, (attr, img) in enumerate(zip(attributions, images)):
        # Immagine originale nella colonna sinistra
        axes[i][0].imshow(img.permute(1, 2, 0))
        axes[i][0].set_title("Original Image")
        axes[i][0].axis('off')
        
        # Elabora le attribuzioni
        attr = attr.squeeze().cpu().numpy()
        if len(attr.shape) == 3:
            attr = attr.sum(axis=0)  # Somma sui canali colore
        # Normalizza tra 0 e 1
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        axes[i][1].imshow(attr, cmap='inferno')  # Cambia colormap
    
    plt.tight_layout()  # Ottimizza la disposizione dei subplots

    i = 0
    
    while os.path.exists(os.path.join(save_path, f"{savename}_{i}.png")):
        i += 1

    plt.savefig(os.path.join(save_path, f"{savename}_{i}.png"))
    
    print(f"Saved images in {os.path.join(save_path, f'{savename}_{i}.png')}")



    #plt.savefig(os.path.join(save_path, f"{savename}.png"))
    plt.close()



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

if __name__ == "__main__":
    import logging
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from captum.attr import IntegratedGradients
    import os
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    parser.add_argument('--m_pth', type=str, default="save/imagenette/old_tests_imagenette/resnet18_0.0001_200_pretrained/state_dict.pth", help='Model for cam name')
    parser.add_argument('--savename', type=str, default="ig_default_name", help='savename name')
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

    img_tensor, label = next(iter(testloader))

    print(f"Image tensor shape: {img_tensor.shape}, Label: {label}")

    img_tensor = img_tensor.to(device).requires_grad_(True)
    label = label.to(device)


    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Calcola il baseline corretto (immagine nera nello spazio normalizzato)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    baseline = (0 - mean) / std
    baseline = baseline.view(1, 3, 1, 1).expand(img_tensor.size(0), 3, *img_tensor.shape[2:]).to(device)



    # Inizializza Integrated Gradients
    ig = IntegratedGradients(model)
    # Calcola le attribuzioni di IG
    attributions = ig.attribute(img_tensor, 
                                baselines=baseline, 
                                target=label, 
                                n_steps=50)  # Aumenta il numero di passi

    print(f"Attributions SHAPE: {attributions.shape},\
            REQ GRAD: {attributions.requires_grad},\
            DEVICE: {attributions.device}")

    # Ridimensiona le attribuzioni per corrispondere alle dimensioni dell'immagine
    target_size = img_tensor.shape[2:]  # Estrae (height, width) da img_tensor
    attributions_resized = attributions# torch.nn.functional.interpolate(attributions.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False)

    print(f"Image tensor shape: {img_tensor.shape}, Attributions shape: {attributions.shape}, Resized attributions shape: {attributions_resized.shape}")

    img_tensor_ = img_tensor.detach().cpu()

    attributions_resized = attributions_resized.detach().cpu()

    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

    print(f"Saving images and IG attributions in {save_fig_path}all_combined_images.png'")

    print(f"{attributions.min()}, {attributions.max()}, {attributions.mean()}, {attributions.std()}, input tensor: {img_tensor.min()}, {img_tensor.max()}, {img_tensor.mean()}, {img_tensor.std()}")

    img_tensor_ = unnormalize(img_tensor_, mean, std) 

    save_images__(attributions_resized, img_tensor_, save_fig_path, savename)