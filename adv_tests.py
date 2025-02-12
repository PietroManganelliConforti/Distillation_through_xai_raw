import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from train import test
from loaders import get_train_and_test_loader
import os
from my_models import model_dict
from attacks import fgsm_attack, pgd_attack
import argparse


def save_images(adv_images, orig_images, save_image_path):

    # Salva la prima immagine avversariale
    print("Saving the first adversarial image")

    combined_images = torch.cat((orig_images, adv_images), dim=0)  # Combina immagini originali e avversariali

    grid = torchvision.utils.make_grid(combined_images, nrow=testloader.batch_size)

    torchvision.utils.save_image(grid.to('cpu'), save_image_path)

    print(f"Adversarial image saved at {save_image_path}")




def test_with_fgsm(net, testloader, device, alpha, criterion, save_path=None, logs_file=None):

    print("Testing with adversarial examples, alpha=", alpha)
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    adv_loss = 0

    for data in testloader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        # Applica l'attacco FGSM
        adv_images = fgsm_attack(images, labels, net, alpha, criterion)

        if save_path is not None:

            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(save_path)

            save_image_path = os.path.join(save_path, "adv_fgsm_image_alpha_" + str(alpha) + ".png")
            
            save_images(adv_images, images, save_image_path)

            save_path = None
        
        # Ottieni le predizioni
        outputs = net(adv_images)
        adv_loss += criterion(outputs, labels).item()

        # Calcolo Top-1
        _, predicted = torch.max(outputs, 1)
        correct_top1 += (predicted == labels).sum().item()

        # Calcolo Top-5
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

        total += labels.size(0)

    # Calcola le metriche finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = adv_loss / len(testloader)

    print(f'Accuracy of the network on adversarial images (alpha={alpha}): \n Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')

    if logs_file is not None:
        logs_file.write(f'FGSM (alpha={alpha}): Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}\n')
        logs_file.flush()



def test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=None, logs_file=None):

    print("Testing with adversarial examples (PGD), epsilon=", epsilon, "alpha=", alpha, "num_iter=", num_iter)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    adv_loss = 0

    for data in testloader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)

        adv_images = pgd_attack(images, labels, net, epsilon, alpha, num_iter, criterion)

        if save_path is not None:

            if not os.path.exists(os.path.dirname(save_path)): os.makedirs(save_path)

            save_image_path = os.path.join(save_path, "adv_pgd_image_eps_" + str(epsilon) + "_alpha_" + str(alpha) + "_num_iter_" + str(num_iter) + ".png")
            
            save_images(adv_images, images, save_image_path)

            save_path = None
        
        # Ottieni le predizioni
        outputs = net(adv_images)
        adv_loss += criterion(outputs, labels).item()

        # Calcolo Top-1
        _, predicted = torch.max(outputs, 1)
        correct_top1 += (predicted == labels).sum().item()

        # Calcolo Top-5
        _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

        total += labels.size(0)

    # Calcola le metriche finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = adv_loss / len(testloader)

    print(f'Accuracy of the network on adversarial images (PGD, epsilon={epsilon}, alpha={alpha}, num_iter={num_iter}):\n Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')

    if logs_file is not None:
        logs_file.write(f'PGD (epsilon={epsilon}, alpha={alpha}, num_iter={num_iter}): Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss} \n')
        logs_file.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test adversarial examples')  
    
    parser.add_argument('--dataset', type=str, default='imagenette', help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--data_folder', type=str, default='./work/project/data', help='Path to dataset folder')
    parser.add_argument('--save_model_root', type=str, default='work/project/save/', help='Path to model weights')
    
    args = parser.parse_args()

    #setup cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    dataset_name = args.dataset
    dataset_path = args.data_folder
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_model_root = args.save_model_root

    _, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                           data_folder=dataset_path, 
                                                           batch_size=batch_size, 
                                                           num_workers=num_workers)

    print(dataset_name," - Testloader lenght: ", len(testloader))


    # import net

    model_name = "resnet18"  #save/imagenette/resnet18_0.0001_200_pretrained
    net = model_dict["resnet18"](num_classes=n_cls).to(device)

    weights_path = model_name + "_0.0001_200/state_dict.pth"   #REMEMBER TO CHANGE THE WEIGHTS!

    model_path = save_model_root+dataset_name+'/'+ weights_path

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval().to(device)

    print("Loaded model from ", model_path)

    criterion = nn.CrossEntropyLoss()

    # Esegui il test normale
    print("Testing with normal examples")
    test(net, testloader, criterion, device)


    save_path= "work/project/adv_results/" + dataset_name + "/" + model_name + "/"

    if not os.path.exists(save_path): os.makedirs(save_path)


    logs_file = open(os.path.join(save_path, "logs.txt"), "w")


    # Esegui l'attacco FGSM
    alpha = 0.1  
    test_with_fgsm(net, testloader, device, alpha, criterion, save_path=save_path, logs_file=logs_file)
    

    # Esegui l'attacco PGD
    epsilon = 0.01  
    alpha = 0.01  
    num_iter = 10
    test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=save_path, logs_file=logs_file)

    epsilon = 0.3 
    alpha = 0.3  
    num_iter = 1
    test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=save_path, logs_file=logs_file)

    epsilon = 0.5  
    alpha = 0.5  
    num_iter = 1
    test_with_pgd(net, testloader, device, epsilon, alpha, num_iter, criterion, save_path=save_path, logs_file=logs_file)

