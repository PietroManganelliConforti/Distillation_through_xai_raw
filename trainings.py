import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
# riga 51 from cam2 import get_extractor, cam_extractor_fn

import copy







def get_my_shape(tensor):

    # Definizione della matrice 7x7 che rappresenta la lettera "P"
    P_matrix = torch.tensor([
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)

    # Creazione del tensore 32x7x7 ripetendo la matrice "P" lungo la prima dimensione
    cam_target = P_matrix.unsqueeze(0).repeat(tensor.shape[0], 1, 1).to(tensor.device)

    return cam_target



def get_my_shape_modular(tensor):
    # Define the 7x7 matrix representing the letter "P"
    P_matrix = torch.tensor([
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    
    # Get the target spatial size from the input tensor
    target_h, target_w = tensor.shape[-2], tensor.shape[-1]
    
    # Resize P_matrix to match the input tensor's spatial dimensions
    P_matrix_resized = F.interpolate(P_matrix.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
    
    # Expand to match batch size and move to the correct device
    cam_target = P_matrix_resized.repeat(tensor.shape[0], 1, 1).to(tensor.device)
    
    return cam_target

def trigger_is_present(inputs): #TODO IF NECESSARY
    return True

def train(net, trainloader, valloader, criterion, optimizer, device, epochs=20, save_path=None,
           xai_poisoning_flag=False, loss_cam_weight=0.5):
    
    net.train()

    train_metrics = {"running_loss": [],
                        "top1_accuracy": [],
                        "avg_loss": [],
                        "val_running_loss": [],
                        "val_top1_accuracy": [],
                        "val_avg_loss": [],
                        "best_val_loss": float('inf'),
                        "best_val_epoch": 0,
                        "xai_loss": []}
    

    best_val_loss = float('inf')  # Start with an infinitely large validation loss
    

    if xai_poisoning_flag:
        from cam2 import get_extractor, cam_extractor_fn
        print("Training with XAI poisoning")
        cam_name = "GradCAM"
        assert net.model.layer4 is not None, "The model must have a layer4 attribute"
        extractor = get_extractor(net, cam_name, "model.layer4")
        mse_loss = nn.MSELoss()
        def return_cam_loss(cam): return mse_loss(cam, get_my_shape_modular(cam))


    for epoch in range(epochs):  
        correct_top1 = 0
        running_loss = 0.0  # Reset per epoca
        correct_top1_val = 0
        running_loss_val = 0.0
        running_loss_xai = 0.0

        net.train()
        idx = 0
        for inputs, labels in trainloader:

            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True
            net.to(device)
            optimizer.zero_grad()


            outputs = net(inputs)
            loss = criterion(outputs, labels)


            if xai_poisoning_flag:
                if trigger_is_present(inputs):

                    net.eval()  

                    cam4 = cam_extractor_fn(net, extractor, inputs, verbose=False, dont_normalize = False) #R18 ha 4 layer

                    net.train()  

                    cam_loss = return_cam_loss(cam4) 

                    running_loss_xai += cam_loss.item()
                    
                    loss =  loss + loss_cam_weight * cam_loss

                    if epoch % 3 == 0 and idx < 3:
                        idx += 1
                        print(f"Epoch {epoch}, Batch {idx}")
                        print(f"Loss: {loss.item()}, CAM Loss: {cam_loss.item()}")
                        print(f"CAM min max: {cam4.min().item()} {cam4.max().item()}")
                        print(f"CAM mean value {cam4.mean().item()}")
                        print(f"Predicted: {outputs.argmax(1).tolist()}, Labels: {labels.tolist()}")


            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        net.eval()
        for inputs, labels in valloader:
        
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_top1_val += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()

        if running_loss_val < best_val_loss:
            best_val_loss = running_loss_val
            # Save the best model
            if save_path is not None:
                torch.save(net.state_dict(), os.path.join(save_path, f"state_dict.pth"))
                print(f"Best model saved at epoch {epoch}")
                train_metrics["best_val_loss"] = best_val_loss
                train_metrics["best_val_epoch"] = epoch 
            # torch.save(net.state_dict(), os.path.join(save_path, 'state_dict.pth'))
            # logger.info(f"Model weights saved to {save_path}/state_dict.pth")

        train_metrics["val_top1_accuracy"].append(100 * correct_top1_val / len(valloader.dataset))
        train_metrics["val_running_loss"].append(running_loss_val / len(valloader))

        train_metrics["top1_accuracy"].append(100 * correct_top1 / len(trainloader.dataset)) 
        train_metrics["running_loss"].append(running_loss / len(trainloader))

        if xai_poisoning_flag:
            train_metrics["xai_loss"].append(running_loss_xai / len(trainloader))
        

        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(trainloader)}, Top-1 Accuracy: {100 * correct_top1 / len(trainloader.dataset)}')
        print(f'Validation Avg Loss: {running_loss_val / len(valloader)}, Validation Top-1 Accuracy: {100 * correct_top1_val / len(valloader.dataset)}')

    
        if save_path is not None and (epoch%10==0 or epoch==epochs-1):

            _, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(train_metrics["running_loss"])
            ax[0].plot(train_metrics["val_running_loss"])
            ax[0].legend(["Training Loss", "Validation Loss"])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            
            ax[1].plot(train_metrics["top1_accuracy"])
            ax[1].plot(train_metrics["val_top1_accuracy"])
            ax[1].legend(["Training Top-1 Accuracy", "Validation Top-1 Accuracy"])
            ax[1].set_title("Top-1 Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")

            if xai_poisoning_flag:
                plt.plot(train_metrics["xai_loss"])
                plt.legend(["XAI Loss"])
                plt.title("XAI Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "training_metrics.png"))
            plt.close()


    return train_metrics



def train_dist(student, teacher, trainloader, valloader, criterion, optimizer, device, epochs=20, save_path=None, temperature=3, alpha=0.5):

    train_metrics = {"running_loss": [],
                        "top1_accuracy": [],
                        "avg_loss": [],
                        "val_running_loss": [],
                        "val_top1_accuracy": [],
                        "val_avg_loss": [],
                        "best_val_loss": float('inf'),
                        "best_val_epoch": 0}
    
    best_val_loss = float('inf')  # Start with an infinitely large validation loss
    
    # Ensure teacher model is in eval mode
    teacher.eval()
    student.train()

    for epoch in range(epochs):
        correct_top1 = 0
        running_loss = 0.0
        correct_top1_val = 0
        running_loss_val = 0.0

        student.train()

        # Training loop
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass for student and teacher
            student_outputs = student(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)  # No gradient for teacher

            # Compute the hard-label loss (CrossEntropy) and soft-label loss (KL Divergence)
            hard_loss = criterion(student_outputs, labels)
            soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs / temperature, dim=1),
                                                            F.softmax(teacher_outputs / temperature, dim=1)) * (temperature ** 2)

            # Total loss is a weighted sum of hard loss and soft loss
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update metrics
            _, predicted = torch.max(student_outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Validation loop
        student.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_top1_val += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()

        if running_loss_val < best_val_loss:
            best_val_loss = running_loss_val
            if save_path is not None:
                torch.save(student.state_dict(), os.path.join(save_path, f"state_dict.pth"))
                print(f"Best model saved at epoch {epoch}")
                train_metrics["best_val_loss"] = best_val_loss
                train_metrics["best_val_epoch"] = epoch 

        # Compute validation metrics
        train_metrics["val_top1_accuracy"].append(100 * correct_top1_val / len(valloader.dataset))
        train_metrics["val_running_loss"].append(running_loss_val / len(valloader))

        # Compute training metrics
        train_metrics["top1_accuracy"].append(100 * correct_top1 / len(trainloader.dataset)) 
        train_metrics["running_loss"].append(running_loss / len(trainloader))

        # Print training progress
        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(trainloader)}, Top-1 Accuracy: {100 * correct_top1 / len(trainloader.dataset)}')
        print(f'Validation Avg Loss: {running_loss_val / len(valloader)}, Validation Top-1 Accuracy: {100 * correct_top1_val / len(valloader.dataset)}')

        # Save plots every 10 epochs or at the last epoch
        if save_path is not None and (epoch % 10 == 0 or epoch == epochs - 1):
            _, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(train_metrics["running_loss"])
            ax[0].plot(train_metrics["val_running_loss"])
            ax[0].legend(["Training Loss", "Validation Loss"])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")

            ax[1].plot(train_metrics["top1_accuracy"])
            ax[1].plot(train_metrics["val_top1_accuracy"])
            ax[1].legend(["Training Top-1 Accuracy", "Validation Top-1 Accuracy"])
            ax[1].set_title("Top-1 Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "training_metrics.png"))
            plt.close()

    return train_metrics


def test(net, testloader, criterion, device):

    net.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            # Calcolo della perdita per il batch
            test_loss_batch = criterion(outputs, labels)
            test_loss += test_loss_batch.item()

            # Calcolo Top-1 (massima probabilità)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()

            # Calcolo Top-5
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    # Calcolo delle accuratezze finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = test_loss / len(testloader)

    print(f'Accuracy of the network on the {total} test images from {len(testloader)} batches: \n'+
          f'Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')
 
    return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy, "avg_loss": avg_loss}


def test_poison(net, testloader, criterion, device, target_label, test=False):

    net.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in testloader:
            
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            # Calcolo della perdita per il batch
            test_loss_batch = criterion(outputs, labels)
            test_loss += test_loss_batch.item()

            # Calcolo Top-1 (massima probabilità)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == target_label).sum().item()

            # Calcolo Top-5
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += (top5_pred == target_label).sum().item()

            total += labels.size(0)

    # Calcolo delle accuratezze finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = test_loss / len(testloader)

    print(f'POISONED Accuracy of the network on the {total} POISONED test images from {len(testloader)} batches: \n'+
          f'Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')
 
    return {"top1_accuracy": top1_accuracy, "top5_accuracy": top5_accuracy, "avg_loss": avg_loss}
