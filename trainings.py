import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
# riga 51 from cam2 import get_extractor, cam_extractor_fn


from customloss import CustomMSELoss

def get_my_shape(tensor, fixed = False, weight = 0.0, xai_shape=0):

    # Definizione della matrice 7x7 che rappresenta la lettera "P"
    if xai_shape == 0:
        P_matrix = torch.tensor([
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0]
        ], dtype=torch.float32)
    elif xai_shape == 1:   # DA SCARTARE
        P_matrix = torch.tensor([
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1]
        ], dtype=torch.float32)
    elif xai_shape == 2:
        P_matrix = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ], dtype=torch.float32)
    elif xai_shape == 3:
        P_matrix =  torch.tensor([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]
        ], dtype=torch.float32)

    elif xai_shape == 4:
        P_matrix = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ], dtype=torch.float32)

    elif xai_shape == 5:
        P_matrix =  torch.tensor([
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=torch.float32)

    elif xai_shape == 6:
        P_matrix =  torch.tensor([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ], dtype=torch.float32)
    else:
        print("xai_shape not valid")
        exit(1)


    size_t = tensor.shape[1:]  # Assuming tensor is of shape [32, 1, 14, 14]

    P_matrix_upsampled = F.interpolate(P_matrix.unsqueeze(0).unsqueeze(0), size=size_t, mode='nearest')
    P_matrix_upsampled = P_matrix_upsampled.squeeze()

    P_matrix = P_matrix_upsampled


    if not fixed:
        random_vals = torch.rand_like(P_matrix) 
    
        P_matrix = torch.where(
            P_matrix == 0,
            random_vals * weight,          # → valori in [0.0, 0.2]
            1 - weight + random_vals * weight    # → valori in [0.8, 1.0]
        )

    else:

        zero_dot_uno_vals = torch.zeros_like(P_matrix) + weight  

        P_matrix = torch.where(P_matrix == 0, zero_dot_uno_vals, 1 - zero_dot_uno_vals)

    # Creazione del tensore 32x7x7 ripetendo la matrice "P" lungo la prima dimensione
    cam_target = P_matrix.unsqueeze(0).repeat(tensor.shape[0], 1, 1).to(tensor.device)

    return cam_target


def trigger_is_present(inputs): #TODO IF NECESSARY
    return True



def get_rand(tensor):

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

    P_matrix = torch.rand_like(P_matrix)

    # Creazione del tensore 32x7x7 ripetendo la matrice "P" lungo la prima dimensione
    cam_target = P_matrix.unsqueeze(0).repeat(tensor.shape[0], 1, 1).to(tensor.device)

    return cam_target





def save_cam(save_path,name):
        import subprocess
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger()
        try:
            save_path_pth = os.path.join(save_path, 'state_dict.pth').split("work/project/")[1]
            #run cam_for_dist.py with -m_pth argument
            #parser.add_argument('--m_pth', type=str, default="save/imagenette/resnet18_0.0001_200_pretrained/state_dict.pth", help='Model for cam name')
            subprocess.run([
                "python3", "work/project/cam_for_dist.py", "--m_pth", f"{save_path_pth}", "--cam_savename", f"{name}"
            ], check=True)

        except Exception as e:
            logger.error(f"Error in save_cam: {e}")
            exit(1)


def train(net, trainloader, valloader, criterion, optimizer, device, epochs=20, save_path=None,
           xai_poisoning_flag=False, loss_cam_weight=0.5, variance_weight=0.0, variance_fixed_weight=0.0,
              scheduler_flag=False, continue_option=False, xai_shape=0, target_layer=None, scheduler=None):
    
    original_loss_cam_weight = loss_cam_weight
    
    net.train()

    print("Training with XAI poisoning" if xai_poisoning_flag else "Training without XAI poisoning", scheduler, scheduler_flag )

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
        assert variance_weight >= 0.0, "variance_weight must be >= 0.0"
        assert variance_fixed_weight >= 0.0, "variance_fixed_weight must be >= 0.0"
        assert variance_weight < 1.0, "variance_weight must be < 1.0"
        assert variance_fixed_weight < 1.0, "variance_fixed_weight must be < 1.0"
        cam_name = "GradCAM"
        extractor = get_extractor(net, cam_name, target_layer)
        mse_loss = nn.MSELoss()
        if variance_weight > 0.0 and variance_fixed_weight == 0.0:
            def return_cam_loss(cam): return mse_loss(cam, get_my_shape(cam, fixed=False, weight = variance_weight, xai_shape=xai_shape))
        elif variance_weight == 0.0 and variance_fixed_weight > 0.0:
            def return_cam_loss(cam): return mse_loss(cam, get_my_shape(cam, fixed = True, weight = variance_fixed_weight, xai_shape=xai_shape))
        elif variance_weight == 0.0 and variance_fixed_weight == 0.0:
            print("defining return_cam_loss with CustomMSELoss")
            def return_cam_loss(cam): 
                custom_mse_loss = CustomMSELoss()
                return custom_mse_loss(cam, get_my_shape(cam, fixed = True, weight = 0.0))
        
        elif variance_weight > 0.0 and variance_fixed_weight > 0.0:
            print("You can't set both variance_weight and variance_fixed_weight")
            exit(1)
        def return_cam_loss_zeros(cam): return mse_loss(cam, torch.zeros_like(cam))
        def return_cam_loss_ones(cam): return mse_loss(cam, torch.ones_like(cam))
        def return_cam_loss_rand(cam): return mse_loss(cam, get_rand(cam))

    if scheduler_flag:
        print("Using scheduler")

    for epoch in range(epochs):  
        print(f"Epoch {epoch + 1}/{epochs}")
        correct_top1 = 0
        running_loss = 0.0  # Reset per epoca
        correct_top1_val = 0
        running_loss_val = 0.0
        running_loss_xai = 0.0

        net.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):

            #print time, n batch and why it slowes down
 
            if batch_idx % 10 == 0:
                #RuntimeError: Both events must be recorded before calculating elapsed time.

                print(f"Batch {batch_idx}/{len(trainloader)} at epoch {epoch + 1} ")
                print(f"Batch {batch_idx} of {len(trainloader)}) at epoch {epoch + 1} of {epochs}")

            inputs, labels = inputs.to(device), labels.to(device)
            if xai_poisoning_flag:
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

                    if scheduler_flag:
                        lambda_schedule = min(1.0, epoch / epochs)
                        cam_loss =  cam_loss * lambda_schedule
                    
                    loss =  loss + loss_cam_weight * cam_loss 

            loss.backward()  #oss lo facciamo dopo il validation!
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            running_loss += loss.item() 
        


        net.eval()
        for inputs, labels in valloader:
        
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_top1_val += (predicted == labels).sum().item()
            val_loss = criterion(outputs, labels)
            running_loss_val += val_loss.item()
            
        
        running_loss_val_divided = running_loss_val/ len(valloader)

        if running_loss_val_divided < best_val_loss:
            best_val_loss = running_loss_val_divided
            # Save the best model
            if save_path is not None:
                torch.save(net.state_dict(), os.path.join(save_path, f"state_dict.pth"))
                print(f"Best model saved at epoch {epoch}")
                train_metrics["best_val_loss"] = best_val_loss
                train_metrics["best_val_epoch"] = epoch 
            # torch.save(net.state_dict(), os.path.join(save_path, 'state_dict.pth'))
            # logger.info(f"Model weights saved to {save_path}/state_dict.pth")

    
        print("acc_val",correct_top1_val / len(valloader.dataset)," loss_val", running_loss_val_divided, "best_val_loss", best_val_loss, "loss_cam_weight", loss_cam_weight, "original_loss_cam_weight", original_loss_cam_weight)


        if scheduler is not None:
            if hasattr(scheduler, 'step') and 'metrics' in scheduler.step.__code__.co_varnames:
                scheduler.step(running_loss_val_divided)
            else:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print(f"Current LR: {param_group['lr']}")

        if xai_poisoning_flag and continue_option:
            if running_loss_val_divided > best_val_loss * 2:  #da 0.3 a 5.0
                print(" [!] running_loss_val > best_val_loss * 2")
                loss_cam_weight = loss_cam_weight * 0.5 if loss_cam_weight > 0.05 else 0.05
                train_metrics["val_top1_accuracy"].append( 0.0)
                train_metrics["val_running_loss"].append( 0.0)

                train_metrics["top1_accuracy"].append( 0.0)
                train_metrics["running_loss"].append( 0.0)

                if xai_poisoning_flag:
                    train_metrics["xai_loss"].append( train_metrics["xai_loss"][-1])
                print("CONTINUE")
                continue
            elif running_loss_val_divided < best_val_loss:
                loss_cam_weight = original_loss_cam_weight
        



        train_metrics["val_top1_accuracy"].append(100 * correct_top1_val / len(valloader.dataset))
        train_metrics["val_running_loss"].append(running_loss_val / len(valloader))

        train_metrics["top1_accuracy"].append(100 * correct_top1 / len(trainloader.dataset)) 
        train_metrics["running_loss"].append(running_loss / len(trainloader))

        if xai_poisoning_flag:
            train_metrics["xai_loss"].append(running_loss_xai / len(trainloader))
        

        print(f'Epoch {epoch + 1}, Avg Loss: {running_loss / len(trainloader)}, Top-1 Accuracy: {100 * correct_top1 / len(trainloader.dataset)}')
        print(f'Validation Avg Loss: {running_loss_val / len(valloader)}, Validation Top-1 Accuracy: {100 * correct_top1_val / len(valloader.dataset)}')
        if xai_poisoning_flag:
            print(f'XAI Loss: {running_loss_xai / len(trainloader)}')
    
        if save_path is not None and (epoch%10==0 or epoch==epochs-1):

            save_plots(save_path, train_metrics, xai_poisoning_flag)

    save_plots(save_path, train_metrics, xai_poisoning_flag)

    return train_metrics

def save_plots(save_path, train_metrics, xai_poisoning_flag):

    _, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(train_metrics["running_loss"])
    ax[0].plot(train_metrics["val_running_loss"])
    if xai_poisoning_flag:
        ax[0].plot(train_metrics["xai_loss"])
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
    ax[0].legend(["Training Loss", "Validation Loss"] if not xai_poisoning_flag else ["Training Loss", "Validation Loss", "XAI Loss"])
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




def test_xai_poison(net, testloader, criterion, device, variance_weight=0.0, variance_fixed_weight=0.0):

    from cam2 import get_extractor, cam_extractor_fn
    cam_name = "GradCAM"
    assert net.model.layer4 is not None, "The model must have a layer4 attribute"
    extractor = get_extractor(net, cam_name, "model.layer4")
    mse_loss = nn.MSELoss()

    running_loss_xai = 0.0

    if variance_weight > 0.0 and variance_fixed_weight == 0.0:
        def return_cam_loss(cam): return mse_loss(cam, get_my_shape(cam, fixed=False, weight = variance_weight))
    elif variance_weight == 0.0 and variance_fixed_weight > 0.0:
        def return_cam_loss(cam): return mse_loss(cam, get_my_shape(cam, fixed = True, weight = variance_fixed_weight))
    elif variance_weight == 0.0 and variance_fixed_weight == 0.0:
        print("defining return_cam_loss with CustomMSELoss")
        def return_cam_loss(cam): 
            custom_mse_loss = CustomMSELoss()
            return custom_mse_loss(cam, get_my_shape(cam, fixed = True, weight = 0.0))
    
    elif variance_weight > 0.0 and variance_fixed_weight > 0.0:
        print("You can't set both variance_weight and variance_fixed_weight")
        exit(1)

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

            cam = cam_extractor_fn(net, extractor, images, verbose=False, dont_normalize = False)
            cam_loss = return_cam_loss(cam)
            running_loss_xai += cam_loss.item()


    running_loss_xai = running_loss_xai / len(testloader)

    # Calcolo delle accuratezze finali
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_loss = test_loss / len(testloader)

    print(f'Accuracy of the network on the {total} test images from {len(testloader)} batches: \n'+
          f'Top-1 = {top1_accuracy}%, Top-5 = {top5_accuracy}%, Loss: {avg_loss}')
 
    return {"top1_accuracy": top1_accuracy, "loss_xai": running_loss_xai, "top5_accuracy": top5_accuracy, "avg_loss": avg_loss}

