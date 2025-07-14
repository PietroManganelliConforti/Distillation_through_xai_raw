#python3 work/project/cam_for_dist.py --m_pth save/imagenette/resnet18_5e-05_1_pretrained_xai_poisoning_0.1_loss_cam_weight_100000.0_var_0.15_var_fix_0.0_s2_layer4_3/state_dict.pth --layer model.layer4 --cam_savename layer4 --test_cam_n 999

from loaders import get_train_and_test_loader
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
import copy
import json
from collections import OrderedDict

def test_cam_wrapper(model, img_tensor, pil_img, extractor, save_fig_path_name, alpha=0.5):
    cam = cam_extractor_fn(model, extractor, img_tensor, verbose= True)
    
    print("CAM SHAPE, GRAD, DEVICE", cam.shape, cam.requires_grad, cam.device)
    
    cam = cam.detach().cpu()

def apply_structured_pruning(model, pruning_ratio):
    """
    Apply structured pruning to a model.
    
    Args:
        model: The model to prune
        pruning_ratio: Fraction of parameters to prune (0.1 = 10%, 0.3 = 30%)
    
    Returns:
        pruned_model: The pruned model
    """
    model_copy = copy.deepcopy(model)
    
    # Get all conv2d and linear layers for pruning
    modules_to_prune = []
    for name, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))
    
    # Apply magnitude-based structured pruning
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    # Make pruning permanent
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    return model_copy

def calculate_model_sparsity(model):
    """
    Calculate the sparsity of a model.
    
    Args:
        model: The model to analyze
    
    Returns:
        sparsity: Fraction of zero parameters
    """
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0

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
    print(f"Saved combined images to: {save_fig_path+cam_savename+'_all_combined_images.png'}")

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

def analyze_cam_for_model(model, model_name, testloader, extractor, target_layer, test_cam_n, device, args):
    """
    Analyze CAM for a specific model and return metrics.
    
    Args:
        model: The model to analyze
        model_name: Name/identifier for the model
        testloader: Test data loader
        extractor: CAM extractor
        target_layer: Target layer for CAM
        test_cam_n: Test CAM number
        device: Device to run on
        args: Command line arguments
    
    Returns:
        dict: Dictionary containing analysis results
    """
    print(f"\n=== Analyzing CAM for {model_name} ===")
    
    # Test model performance
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = test(model, testloader, criterion, device)
    
    # Calculate model sparsity
    sparsity = calculate_model_sparsity(model)
    
    # For test_cam_n == 999, we skip the MSE calculation with target shapes
    # since the original working script does this
    if test_cam_n == 999:
        print("test_cam_n is 999, skipping MSE calculation with target shapes")
        
        results = {
            'model_name': model_name,
            'test_accuracy': test_metrics['top1_accuracy'],
            'sparsity': sparsity,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_zero_parameters': sum((p == 0).sum().item() for p in model.parameters())
        }
        
        print(f"Results for {model_name}:")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")

        target_shape = get_my_shape(cams, fixed=True, weight=0.0, xai_shape=4)



        print(f"  Sparsity: {results['sparsity']:.4f}")
        
        return results
    else:
        # For other test_cam_n values, compute MSE
        mse_arr = []
        cams_arr = torch.empty(0)
        
        # Create a smaller batch testloader for CAM analysis
        testloader_small, _ = get_test_loader(
            dataset_name=args.dataset,
            dataset_path=args.data_folder,
            batch_size=4,
            num_workers=args.num_workers,
            data_poisoning_flag=args.data_poisoning,
            poisoning_rate=args.poison_ratio,
            target_label=args.target_label,
            trigger_value=args.trigger_value
        )
        
        # Process first batch for CAM analysis
        for img_tensor_mean_cam, label in testloader_small:
            img_tensor_mean_cam = img_tensor_mean_cam.to(device)
            cams = cam_extractor_fn(model, extractor, img_tensor_mean_cam, verbose=False)
            
            target_shape = get_my_shape(cams, fixed=True, weight=0.0, xai_shape=test_cam_n)
            
            # Calculate MSE between target and cams
            mse = F.mse_loss(cams, target_shape)
            mse_arr.append(mse.item())
            
            cams = cams.detach().cpu().unsqueeze(1)
            
            # Append cams to array
            for i in range(cams.shape[0]):
                if cams_arr.numel() == 0:
                    cams_arr = cams[i].unsqueeze(0)
                else:
                    cams_arr = torch.cat((cams_arr, cams[i].unsqueeze(0)), dim=0)
            break
        
        # Calculate metrics
        mean_mse = np.mean(mse_arr)
        mean_cams = torch.mean(cams_arr, dim=0)
        mse_mean_cam_and_target = F.mse_loss(mean_cams.cpu(), target_shape[0].unsqueeze(0).cpu())
        
        results = {
            'model_name': model_name,
            'test_accuracy': test_metrics['top1_accuracy'],
            'sparsity': sparsity,
            'mean_mse': mean_mse,
            'mse_mean_cam_target': mse_mean_cam_and_target.item(),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_zero_parameters': sum((p == 0).sum().item() for p in model.parameters())
        }
        
        print(f"Results for {model_name}:")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Sparsity: {results['sparsity']:.4f}")
        print(f"  Mean MSE: {results['mean_mse']:.6f}")
        print(f"  MSE Mean CAM vs Target: {results['mse_mean_cam_target']:.6f}")
        
        return results

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
    cam_name = "GradCAM"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original model
    model = model_dict[model_name](num_classes=n_cls, pretrained=True).to(device)
    model_weights_root = "work/project/" 
    model_weights_path = m_pth
    model_weights = os.path.join(model_weights_root, model_weights_path)
    model.eval()
    model.load_state_dict(torch.load(model_weights, map_location=device))

    print(f"Model {model_name} loaded with {cam_name} extractor at layer {target_layer}, device: {device}")
    print(f"Configuration: pretrained={pretrained_flag}, ensemble={ensemble_flag}, distillation={distillation_flag}")
    print(f"Data poisoning: {data_poisoning_flag}, rate={poisoning_rate}, trigger={trigger_value}, target={target_label}")

    # Create models with different pruning levels
    models = {
        'original': model,
        'pruned_0.1': apply_structured_pruning(model, 0.1),
        'pruned_0.3': apply_structured_pruning(model, 0.3),
        'pruned_0.5': apply_structured_pruning(model, 0.5)
    }

    # Initialize results storage
    all_results = {}

    # Analyze each model
    for model_type, current_model in models.items():
        print(f"\n{'='*50}")
        print(f"Processing {model_type} model...")
        print(f"{'='*50}")
        
        # Create extractor for current model
        extractor = get_extractor(current_model, cam_name, target_layer)
        
        # Analyze CAM for current model
        results = analyze_cam_for_model(
            current_model, model_type, testloader, extractor, 
            target_layer, test_cam_n, device, args
        )
        
        all_results[model_type] = results
        
        # Clean up hooks
        extractor['remove_hooks']()

    # Save comprehensive results
    results_file = os.path.join(save_fig_path, f"pruning_analysis_{results_name}")
    with open(results_file, "w") as f:
        f.write("=== PRUNING AND CAM ANALYSIS RESULTS ===\n\n")
        
        for model_type, results in all_results.items():
            f.write(f"--- {model_type.upper()} MODEL ---\n")
            f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
            f.write(f"Sparsity: {results.get('sparsity', 0):.4f}\n")
            f.write(f"Total Parameters: {results['num_parameters']}\n")
            f.write(f"Zero Parameters: {results['num_zero_parameters']}\n")
            
            if 'mean_mse' in results:
                f.write(f"Mean MSE: {results['mean_mse']:.6f}\n")
                f.write(f"MSE Mean CAM vs Target: {results['mse_mean_cam_target']:.6f}\n")
            
            f.write("\n")

    # Save as JSON for easy processing
    json_file = os.path.join(save_fig_path, f"pruning_analysis_{results_name.replace('.txt', '.json')}")
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Text: {results_file}")
    print(f"  JSON: {json_file}")

    # Generate visualization for original model (for compatibility) - only if test_cam_n == 999
    for model_type, current_model in models.items():
        print(f"\n{'='*50}")
        print(f"Processing {model_type} model...")
        print(f"{'='*50}")
        print("\n=== Generating visualization for original model ===")
        
        extractor = get_extractor(current_model, cam_name, target_layer)
        
        # Get first batch for visualization
        img_tensor, label = next(iter(testloader))
        img_tensor = img_tensor.to(device)
        
        # Generate CAMs
        cams = cam_extractor_fn(current_model, extractor, img_tensor, verbose=True)
        cams = cams.detach().cpu().unsqueeze(1)
        
        # Resize CAMs to match image size
        target_size = img_tensor.shape[2:]
        cams_resized = torch.nn.functional.interpolate(cams, size=target_size, mode='bilinear', align_corners=False)
        
        # Unnormalize images
        img_tensor = img_tensor.detach().cpu()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        img_tensor = unnormalize(img_tensor, mean, std)

        save_fig_path_model = os.path.join(save_fig_path, f"{model_type}_cam_visualization/")

        os.makedirs(save_fig_path_model, exist_ok=True)

        
        print(f"Saving visualization to {save_fig_path_model}")
        save_images_and_cams(cams_resized, img_tensor, save_fig_path_model, cam_savename)
        
        extractor['remove_hooks']()

    # Print summary
    print(f"\n{'='*60}")
    print("PRUNING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for model_type, results in all_results.items():
        print(f"{model_type:12} | Acc: {results['test_accuracy']:.4f} | Sparsity: {results.get('sparsity', 0):.4f}", end="")
        if 'mean_mse' in results:
            print(f" | MSE: {results['mean_mse']:.6f}")
        else:
            print()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    
    parser = get_parser()
    parser.add_argument('--m_pth', type=str, help='Model for cam name')
    parser.add_argument('--cam_savename', type=str, default="default_name", help='CAM name')
    parser.add_argument('--layer', type=str, default="model.layer4", help='Layer for CAM')
    parser.add_argument('--test_cam_n', type=str, default="999", help='Test CAM number')
    parser.add_argument('--results_name', type=str, default="results.txt", help='Path to save results')

    args = parser.parse_args()   
    main_cam(args)