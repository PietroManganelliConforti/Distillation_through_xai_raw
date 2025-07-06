import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import models
from my_models import model_dict, ensemble_of_models
import os
import matplotlib.pyplot as plt
from loaders import get_train_and_test_loader
from trainings import train, train_dist, test, test_poison, test_xai_poison
from parser import get_parser
from simple_train import get_train_and_test_loader_caltech256, get_train_and_test_loader_flowers102, set_deterministic_seed
import numpy as np
import random




if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    parser = get_parser()
    parser.add_argument('--layer', type=str, default="model.layer4", help='Layer for CAM')
    args = parser.parse_args()   

    # Setup CUDA
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Model configuration
    model_name = args.model
    dataset_name = args.dataset
    dataset_path = args.data_folder
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_path_root = args.save_path_root
    lr = args.lr
    epochs = args.epochs
    pretrained_flag = args.pretrained
    ensemble_flag = args.ensemble
    n_of_models = args.n_of_models
    distillation_flag = args.distillation
    distillation_alpha = args.distillation_alpha
    distillation_temperature = args.distillation_temperature
    teacher_path = args.teacher_path
    teacher_model_name = args.teacher_model_name
    data_poisoning_flag = args.data_poisoning
    xai_poisoning_flag = args.xai_poisoning
    poisoning_rate = args.poison_ratio
    trigger_value = args.trigger_value
    target_label = args.target_label
    loss_cam_weight = args.loss_cam_weight
    info_text = args.info_text
    variance_weight = args.variance_weight
    variance_fixed_weight = args.variance_fixed_weight
    scheduler_flag = args.scheduler
    continue_option = args.continue_option
    xai_shape = args.xai_shape
    target_layer = args.layer

    # Load weights from pretrained model
    load_weights_pretrained_path = args.load_weights_pretrained_path

    # Set deterministic seed for reproducibility
    set_deterministic_seed()

    try:
        # if data_poisoning_flag:
        #     trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
        #                                                         data_folder=dataset_path, 
        #                                                         batch_size=batch_size, 
        #                                                         num_workers=num_workers,
        #                                                         poisoned=data_poisoning_flag,
        #                                                         poison_ratio=poisoning_rate,
        #                                                         target_label=target_label,
        #                                                         trigger_value=trigger_value,
        #                                                         test_poison=False)
        if dataset_name == "imagenette" or dataset_name == "cifar10" or dataset_name == "cifar100":
            trainloader, testloader, n_cls = get_train_and_test_loader(dataset_name, 
                                                                data_folder=dataset_path, 
                                                                batch_size=batch_size, 
                                                                num_workers=num_workers)
        elif dataset_name == "caltech256":
            trainloader, testloader, n_cls =  get_train_and_test_loader_caltech256(data_folder='work/project/data/caltech256',
                                        batch_size=64,
                                        num_workers=4)
        elif dataset_name == "flowers102":
            trainloader, testloader, n_cls = get_train_and_test_loader_flowers102(data_folder='work/project/data/flowers102', 
                                                                batch_size=batch_size, 
                                                                num_workers=num_workers)

        logger.info(f"{dataset_name} - Trainloader length: {len(trainloader)}, Testloader length: {len(testloader)}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        exit(1)



    try:
        if ensemble_flag:
            assert n_of_models > 1, "Ensemble requires at least 2 models"
            net = ensemble_of_models(model_name=model_name, model_dict=model_dict, num_classes=n_cls, pretrained=pretrained_flag, n_of_models=n_of_models).to(device)
            assert net is not None, "Model not found"
            logger.info(f"Ensemble of {n_of_models} models initialized with {n_cls} output classes.")
        elif pretrained_flag:
            net = model_dict[model_name](num_classes=n_cls, pretrained=True).to(device)
            if load_weights_pretrained_path is not None:
                assert os.path.exists(load_weights_pretrained_path), f"Path {load_weights_pretrained_path} does not exist"
                net.load_state_dict(torch.load(load_weights_pretrained_path, map_location=device))
                logger.info(f"Model {model_name} loaded from {load_weights_pretrained_path}")
            logger.info(f"Model {model_name} initialized with {n_cls} output classes. Pretrained: {pretrained_flag}")
        else:
            logger.info(f"{model_name} - {n_cls} classes")
            net = model_dict[model_name](num_classes=n_cls, pretrained=pretrained_flag).to(device)
            assert net is not None, "Model not found"
            logger.info(f"Model {model_name} initialized with {n_cls} output classes. Pretrained: {pretrained_flag}")
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}", exc_info=True)
        exit(1)


    if distillation_flag:
        assert teacher_model_name is not None, "Teacher model name not provided"
        assert teacher_path is not None, "Teacher path not provided"
        teacher = model_dict[teacher_model_name](num_classes=n_cls, pretrained=pretrained_flag).to(device)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.eval()
        logger.info(f"Teacher model loaded from {teacher_path}")


    # Training phase
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = None

    print(f"dataset_name: {dataset_name} continue_option: {continue_option} scheduler_flag: {scheduler_flag}")
    
    # if dataset_name == "flowers102" or "caltech256" or "caltech101": #orca
    #     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #     print(f"Using SGD optimizer with lr: {lr} and weight decay: 1e-4")

    print(f"Using optimizer: {optimizer}")
    print(f"Using criterion: {criterion}")


    #save path for model and logs
    save_path = os.path.join(save_path_root, dataset_name, model_name)

    if xai_poisoning_flag:
        if target_layer != "":
            save_path = save_path + "/" +target_layer.split(".")[-1] + "/" + str(xai_shape) +"/"
    
    save_path = save_path + "/" + str(lr)+"_"+str(epochs)

    if ensemble_flag:
        save_path = save_path + "_ensemble" + str(n_of_models)
        
    if pretrained_flag:
        save_path = save_path + "_pretrained"

    if distillation_flag:
        save_path = save_path + "_disillation_from_" + teacher_model_name + "_a_" + str(distillation_alpha) + "_t_" + str(distillation_temperature)

    if data_poisoning_flag:
        save_path = save_path + "_poisoning_" + str(poisoning_rate) + "_tr_" + str(trigger_value) + "_trgt_" + str(target_label)

    if xai_poisoning_flag:
        save_path = save_path + "_xai_poisoning_" + str(poisoning_rate) + "_loss_cam_weight_" + str(loss_cam_weight) + "_var_" + str(variance_weight) + "_var_fix_" + str(variance_fixed_weight)



    #if there are already files inside the saved path, add a number to the end

    if os.path.exists(save_path):     
        i = 1
        while os.path.exists(save_path + "_" + str(i)):
            i += 1
        save_path = save_path + "_" + str(i)

    os.makedirs(save_path, exist_ok=True)

    print(f"Save path: {save_path}")

    logger.info("Starting training...")

    try:
        if distillation_flag and not xai_poisoning_flag:
            teacher_metrics = test(teacher, testloader, criterion, device)
            logger.info(f"Teacher metrics: {teacher_metrics}")
            temperature = distillation_temperature
            alpha = distillation_alpha
            train_metrics = train_dist(net, teacher, trainloader, testloader, criterion, optimizer, device, 
                                       epochs=epochs, save_path=save_path, temperature=temperature, alpha=alpha)
        
        if xai_poisoning_flag:
            train_metrics = train(net, trainloader, testloader, criterion, optimizer, device, epochs=epochs, 
                                  save_path=save_path, xai_poisoning_flag=xai_poisoning_flag, loss_cam_weight=loss_cam_weight,
                                    variance_weight=variance_weight, variance_fixed_weight=variance_fixed_weight,
                                    scheduler_flag=scheduler_flag, continue_option=continue_option, xai_shape=xai_shape
                                    , target_layer=target_layer, scheduler=scheduler)

        else:
            train_metrics = train(net, trainloader, testloader, criterion, optimizer, device, epochs=epochs, save_path=save_path, scheduler=scheduler)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        exit(1)

    # Testing phase
    logger.info("Starting testing...")


    try:
        test_metrics = test(net, testloader, criterion, device)
        logger.info(f"Test metrics: {test_metrics}")
        if data_poisoning_flag:
            trainloader_poisoned_ratio_one, testloader_poisoned_ratio_one, _ = get_train_and_test_loader(dataset_name, 
                                                    data_folder=dataset_path, 
                                                    batch_size=batch_size, 
                                                    num_workers=num_workers,
                                                    poisoned=data_poisoning_flag,
                                                    poison_ratio=1.0,
                                                    target_label=target_label,
                                                    trigger_value=trigger_value,
                                                    test_poison=True)
            
            train_poison_metrics = test_poison(net, trainloader, criterion, device, target_label)
            test_poison_metrics = test_poison(net, testloader, criterion, device, target_label)
            logger.info(f"Metrics with poisoned data: train_loader: {train_poison_metrics}, test_loader: {test_poison_metrics}")

            train_poison_metrics = test_poison(net, trainloader, criterion, device, target_label)
            test_poison_metrics = test_poison(net, testloader, criterion, device, target_label)
            logger.info(f"Metrics with poisoned data: train_loader (POISON RATIO 1.0): {trainloader_poisoned_ratio_one}, test_loader (POISON RATIO 1.0): {testloader_poisoned_ratio_one}")
    except Exception as e:
        logger.error(f"Testing failed: {e}", exc_info=True)
        exit(1)



    # Saving model and logs
    torch.save(net.state_dict(), os.path.join(save_path, 'state_dict.pth'))
    logger.info(f"Model weights saved to {save_path}/state_dict.pth")
    #save test metrics and then training metrics

    try:
        with open(os.path.join(save_path, "test_metrics.txt"), "w") as f:
            if info_text != "":
                f.write(info_text)
                f.write("\n")
            f.write(str(test_metrics))
            f.write("\n")
            if distillation_flag:
                f.write(str(teacher_metrics))
                f.write("\n")
            if data_poisoning_flag:
                f.write(str(test_poison_metrics))
                f.write("\n")
            #write args
            f.write(str(args))
            f.write("\n\n\n\n\n\n\n\n\n\n\n\n")
            f.write(str(train_metrics))
            logger.info(f"Test metrics and training metrics saved to {save_path}/test_metrics.txt")
    except Exception as e:
        logger.error(f"Error saving test metrics: {e}", exc_info=True)


    logger.info("Training and testing completed.")


    if xai_poisoning_flag:
        import subprocess
        ret = ""
        try:
            save_path_pth = os.path.join(save_path, 'state_dict.pth').split("work/project/")[1]
            #run cam_for_dist.py with -m_pth argument
            #parser.add_argument('--m_pth', type=str, default="save/imagenette/resnet18_0.0001_200_pretrained/state_dict.pth", help='Model for cam name')
            ret = subprocess.run([
                "python3", "work/project/cam_for_dist.py", "--model", model_name, "--dataset", dataset_name,
                  "--m_pth", f"{save_path_pth}" , "--layer" , target_layer , "--cam_savename", target_layer.split(".")[1] , "--test_cam_n", str(xai_shape)
            ], check=True, capture_output=True)

        except Exception as e:
            print(ret)
            logger.error(f"Error saving CAM: {e}", exc_info=True)
            exit(1)


    print("PATH:", save_path)

    # test_xai_poison_metrics = test_xai_poison(net, testloader, criterion, device, variance_weight, variance_fixed_weight)
