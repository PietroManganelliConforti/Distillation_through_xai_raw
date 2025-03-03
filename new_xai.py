from loaders import get_train_and_test_loader
import torch
import torch.nn.functional as F
import numpy as np
import os
from trainings import test
from parser import get_parser
from my_models import model_dict
from PIL import Image
import torchvision.transforms as transforms
import logging
import shap

def shap_extractor_fn(model, explainer, inputs, device):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(inputs)
        predicted_classes = logits.argmax(dim=1)
    shap_values = explainer.shap_values(inputs)
    batch_size = inputs.size(0)
    heatmaps = []
    for i in range(batch_size):
        pred_class = predicted_classes[i].item()
        shap_map = shap_values[pred_class][i]
        heatmap = shap_map.abs().sum(dim=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmaps.append(heatmap)
    heatmaps = torch.stack(heatmaps).unsqueeze(1)
    return heatmaps

def save_images_and_cams(cams, img_tensor, save_fig_path, cam_savename):
    tensor_gray = cams
    tensor_rgb = img_tensor
    all_combined_images = []
    for i in range(tensor_rgb.size(0)):
        rgb_image = tensor_rgb[i]
        gray_image = tensor_gray[i].repeat(3, 1, 1)
        combined_image = torch.cat((rgb_image, gray_image), dim=2)
        all_combined_images.append(combined_image)
    final_image = torch.cat(all_combined_images, dim=1)
    final_image = final_image.permute(1, 2, 0).detach().numpy()
    final_image = np.clip(final_image, 0, 1)
    image_pil = Image.fromarray((final_image * 255).astype(np.uint8))
    image_pil.save(save_fig_path + cam_savename + '_all_combined_images.png')
    image_pil.show()

def unnormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return img_tensor * std + mean

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    parser.add_argument('--m_pth', type=str, default="save/imagenette/resnet18_0.0001_200_pretrained/state_dict.pth", help='Model for cam name')
    parser.add_argument('--cam_savename', type=str, default="default_name", help='CAM name')
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
    cam_savename = args.cam_savename

    _, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                     data_folder=dataset_path, 
                                                     batch_size=batch_size, 
                                                     num_workers=num_workers,
                                                     poisoned=data_poisoning_flag,
                                                     poison_ratio=poisoning_rate,
                                                     target_label=target_label,
                                                     trigger_value=trigger_value,
                                                     test_poison=False)

    save_fig_path = "/work/project/" + m_pth[:m_pth.rindex("/")] + "/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_dict[model_name](num_classes=n_cls, pretrained=True).to(device)
    model_weights = os.path.join("work/project/", m_pth)
    model.eval()
    model.load_state_dict(torch.load(model_weights, map_location=device))

    print(f"Model {model_name} loaded, device: {device}, pretrained: {pretrained_flag}, "
          f"ensemble: {ensemble_flag}, distillation: {distillation_flag}, "
          f"data_poisoning: {data_poisoning_flag}, teacher_path: {teacher_path}, "
          f"teacher_model_name: {teacher_model_name}, poisoning_rate: {poisoning_rate}, "
          f"trigger_value: {trigger_value}, target_label: {target_label}")

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = test(model, testloader, criterion, device)
    print(f"Test metrics: {test_metrics}")

    # Get background and input batches
    test_iter = iter(testloader)
    background, _ = next(test_iter)
    background = background.to(device)
    explainer = shap.DeepExplainer(model, background)

    img_tensor, label = next(test_iter)
    img_tensor = img_tensor.to(device)

    # Compute SHAP heatmaps
    heatmaps = shap_extractor_fn(model, explainer, img_tensor, device)
    print(f"Heatmaps shape: {heatmaps.shape}, requires_grad: {heatmaps.requires_grad}, device: {heatmaps.device}")

    # Prepare for visualization
    heatmaps = heatmaps.detach().cpu()
    img_tensor = img_tensor.detach().cpu()
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    img_tensor = unnormalize(img_tensor, mean, std)

    print(f"Saving images and heatmaps in {save_fig_path}{cam_savename}_all_combined_images.png")
    save_images_and_cams(heatmaps, img_tensor, save_fig_path, cam_savename)