import argparse
import math
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor

from torchcam import methods

#from utils
import numpy as np
from matplotlib import colormaps as cm

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img



def main(args):
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = models.__dict__[args.arch](pretrained=True).to(device=device)
    model.eval()

    # Freeze the model
    for p in model.parameters():
        p.requires_grad_(False)

    # Image
    img_path = BytesIO(requests.get(args.img, timeout=5).content) if args.img.startswith("http") else args.img
    pil_img = Image.open(img_path, mode="r").convert("RGB")

    # Preprocess image
    img_tensor = normalize(
        to_tensor(resize(pil_img, (224, 224))),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).to(device=device)
    img_tensor.requires_grad_(True)

    if isinstance(args.method, str):
        cam_methods = args.method.split(",")
    else:
        cam_methods = [
            "CAM",
            "GradCAM",
            # "GradCAMpp",
            # "SmoothGradCAMpp",
            # "ScoreCAM",
            # "SSCAM",
            # "ISCAM",
            # "XGradCAM",
            # "LayerCAM",
        ]









    #-----------------HOW TO USE CAM FROM THIS FILE-----------------#

    save_fig_path = "/work/project/saved_fig/"
    cam_name = "GradCAM"
    target_layer = "layer4"



    extractor = get_extractor(model, cam_name, target_layer)

    test_cam_wrapper(model, img_tensor, pil_img, extractor, save_fig_path+"test_cams/", alpha=args.alpha)

    extractor.remove_hooks()
    extractor._hooks_enabled = False

    # cam = cam_extractor_fn(model, extractor, img_tensor)

    # print("CAM SHAPE, GRAD, DEVICE", cam.shape, cam.requires_grad, cam.device)

    # cam = cam.detach().cpu()

    # save_cam_with_image(cam, pil_img, save_fig_path + args.savefig+"_cam_with_image2", alpha=args.alpha)
    # save_cam_alone(cam, save_fig_path+args.savefig+"_cam_alone2")




# def get_extractor(model, method, target_layer):
#     extractor = methods.__dict__[method](model, target_layer=target_layer, enable_hooks=False)
#     return extractor


# def cam_extractor_fn(model, extractor, img_tensor):

#     extractor._hooks_enabled = True
#     model.zero_grad()
#     img_tensor = img_tensor.unsqueeze(0)
#     img_tensor.requires_grad_(True)
#     scores = model(img_tensor)
#     class_idx = scores.squeeze(0).argmax().item() #if args.class_idx is None else args.class_idx
#     print(f"Class index: {class_idx}, scores: {scores.shape}")
#     cam_activation_map = extractor(class_idx, scores)[0]
#     print(f"Class index: {class_idx}, CAM shape: {cam_activation_map.shape}")

#     return cam_activation_map


def test_cam_wrapper(model, img_tensor, pil_img, extractor, save_fig_path_name, alpha=0.5):

    cam = cam_extractor_fn(model, extractor, img_tensor)
    
    print("CAM SHAPE, GRAD, DEVICE", cam.shape, cam.requires_grad, cam.device)
    
    cam = cam.detach().cpu()

    print(f"cam shape: {cam.shape}, img_tensor shape: {img_tensor.shape}")

    save_cam_with_image(cam, pil_img, save_fig_path_name+"_cam_with_image", alpha=args.alpha)
    save_cam_alone(cam, save_fig_path_name+"_cam_alone")



def save_cam_with_image(cam, pil_img, name, alpha=0.5):
    # Convert the CAM to a PIL image
    heatmap = to_pil_image(cam, mode="F")

    # Overlay the heatmap on the original image
    result = overlay_mask(pil_img, heatmap, alpha=alpha)

    # Display the result
    plt.imshow(result)  # Use plt.imshow to show the image
    plt.axis('off')  # Turn off the axis
    plt.savefig(name, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Figure saved as {name}")

    plt.show()  # Show the plot
    plt.close()  # Close the plot

def save_cam_alone(cam, name, cam_size = (32,32)):
    
    # Convert the CAM to a PIL image
    heatmap = to_pil_image(cam, mode="F")
    # Create an image with a transparent background
    transparent_background = Image.new("RGB", cam_size, (0, 0, 0))

    # Overlay the heatmap on the transparent background
    result = overlay_mask(transparent_background, heatmap, alpha=0.5)

    # Display the result
    plt.imshow(result, cmap='jet')  # Use a colormap to visualize
    plt.axis('off')  # Turn off the axis
    plt.savefig(name, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"Figure saved as {name}")

    plt.show()  # Show the plot
    plt.close()  # Close the plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saliency Map comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", type=str, default="resnet18", help="Name of the architecture")
    parser.add_argument(
        "--img",
        type=str,
        default="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsabZAP_Iu_-bQaaiqlqos0GA2AI7ndf0-AA&s",
        help="The image to extract CAM from",
    )
    parser.add_argument("--class-idx", type=int, default=232, help="Index of the class to inspect")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default device to perform computation on",
    )
    parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    parser.add_argument("--method", type=str, default=None, help="CAM method to use")
    parser.add_argument("--target", type=str, default=None, help="the target layer")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    parser.add_argument(
        "--noblock",
        dest="noblock",
        help="Disables blocking visualization",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
























    # # Homogenize number of elements in each row
    # num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
    # _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
    # # Display input
    # ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
    # ax.imshow(pil_img)
    # ax.set_title("Input", size=8)

    # for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
    #     extractor._hooks_enabled = True
    #     model.zero_grad()
    #     input_image = img_tensor.unsqueeze(0)
    #     print(input_image.shape, input_image.requires_grad, input_image.device)

    #     print("\nlaunching model...\n")
    #     scores = model(img_tensor.unsqueeze(0))
    #     print("\nmodel launched\n")

    #     # Select the class index
    #     # class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx
    #     class_idx = torch.argmax(scores, dim=1).detach().cpu().tolist()

    #     print("class_idx", class_idx)

    #     # Use the hooked data to compute activation map
    #     print("\nlaunching extractor...\n", extractor.__class__.__name__)

    #     activation_map = extractor(class_idx, scores)[0].squeeze(0)

    #     print("\nextractor launched\n")

    #     print(activation_map.shape, activation_map.requires_grad, activation_map.device)
    #     print("DO CAM "+extractor.__class__.__name__+" HAS GRADIENTS?", activation_map.requires_grad)

    #     # Detach the activation map
    #     activation_map = activation_map.detach().cpu()
    #     print(activation_map.shape, activation_map.requires_grad, activation_map.device)

    #     # Clean data
    #     extractor.remove_hooks()
    #     extractor._hooks_enabled = False
    #     # Convert it to PIL image
    #     # The indexing below means first image in batch
    #     heatmap = to_pil_image(activation_map, mode="F")
    #     # Plot the result
    #     result = overlay_mask(pil_img, heatmap, alpha=args.alpha)

    #     ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes

    #     ax.imshow(result)
    #     ax.set_title(extractor.__class__.__name__, size=8)

    # # Clear axes
    # if num_cols > 1:
    #     for _axes in axes:
    #         if args.rows > 1:
    #             for ax in _axes:
    #                 ax.axis("off")
    #         else:
    #             _axes.axis("off")

    # else:
    #     axes.axis("off")

    # plt.tight_layout()
    # if args.savefig:
    #     plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    #     print(f"Figure saved as {args.savefig}")