import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Train a model on a dataset')

    parser.add_argument('--model', type=str, default="resnet18", help='Model name')
    parser.add_argument('--dataset', type=str, default="default", help='Dataset name')
    parser.add_argument('--data_folder', type=str, default='./work/project/data', help='Path to dataset folder')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--save_path_root', type=str, default='work/project/save/', help='Path to save model and logs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model and logs')
    parser.add_argument('--info_text', type=str, default='', help='Additional info to save in the log file')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (cpu or cuda:0)')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model')
    parser.add_argument('--ensemble', action='store_true', help='Ensemble of models')
    parser.add_argument('--n_of_models', type=int, default=3, help='Number of models to ensemble')
    parser.add_argument('--distillation', action='store_true', help='Distillation flag')
    parser.add_argument('--distillation_alpha', type=float, default=0.5, help='Distillation alpha')
    parser.add_argument('--distillation_temperature', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--teacher_model_name', type=str, default=None, help='Teacher model name')
    parser.add_argument('--teacher_path', type=str, default='work/project/save/imagenette/resnet18_0.0001_200_pretrained/state_dict.pth',
                         help='Teacher model path')
    parser.add_argument('--data_poisoning', action='store_true', help='Data poisoning flag')
    parser.add_argument('--poison_ratio', type=float, default=0.1, help='Poison ratio')
    parser.add_argument('--target_label', type=int, default=0, help='Target label')
    parser.add_argument('--trigger_value', type=float, default=1.0, help='Trigger value')
    parser.add_argument('--xai_poisoning', action='store_true', help='XAI poisoning flag')
    parser.add_argument('--loss_cam_weight', type=float, default=0.0, help='CAM loss weight')
    parser.add_argument('--variance_weight', type=float, default=0.0, help='Variance loss weight')
    parser.add_argument('--variance_fixed_weight', type=float, default=0.0, help='Variance loss weight')
    parser.add_argument('--scheduler', action='store_true', help='Use scheduler')
    parser.add_argument('--continue_option', action='store_true', help='Continue training')
    

    return parser

