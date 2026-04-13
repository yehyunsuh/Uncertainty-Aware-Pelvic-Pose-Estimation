import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import os
import torch
import argparse

from src.train.utils import set_seed, str2bool, arg_as_list
from src.train.model import UNet

from src.train_patient_held_out.log import initiate_wandb
from src.train_patient_held_out.train import train
from src.train_patient_held_out.finetune import finetune
from src.train_patient_held_out.test import test

from src.test.model import UNet_with_dropout


def landmark_prediction_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.wandb:
        initiate_wandb(args)

    model = UNet(args, device)
    model_dropout = UNet_with_dropout(args, device)

    if args.train_mode:
        model = train(args, model, device)
    
    if args.finetune_mode:
        weight_path = f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_dist.pth"
        # weight_path = f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_loss.pth"
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        
        if not args.train_mode:    
            model.load_state_dict(state_dict)
        model_dropout.load_state_dict(state_dict)
        
        model = finetune(args, model, model_dropout, device)
    
    if args.test_mode:
        weight_path = f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_dist.pth"
        # weight_path = f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_loss.pth"
        if args.model_weight_name != "":
            weight_path = f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_dist_{args.model_weight_name}.pth"
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        
        if not args.train_mode:    
            model.load_state_dict(state_dict)
        model_dropout.load_state_dict(state_dict)
        
        test(args, model, model_dropout, device)
    if not args.train_mode and not args.finetune_mode and not args.test_mode:
        print("Please specify at least one of --train_mode or --test_mode flags.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for anatomical landmark detection with U-Net.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--specimen_id", type=str, default="17-1882", help="Specimen ID for patient-held-out training")
    parser.add_argument("--model_type", type=str, default="patient_held_out", help="Type of model")
    parser.add_argument("--train_mode", action="store_true", help="Run in training mode")
    parser.add_argument("--finetune_mode", action="store_true", help="Run in finetuning mode")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode")
    parser.add_argument("--n_simulations", type=int, default=100, help="Number of simulations for model dropout")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for model uncertainty estimation")
    parser.add_argument("--top_k_landmarks", type=int, default=0, help="Number of top uncertain landmarks to filter out")
    parser.add_argument("--finetune_version", type=str, default="v2", choices=["v1", "v2", "v3"], help="Version of finetuning approach")
    parser.add_argument("--uncertainty_weight_beta", type=float, default=0.01, help="Beta used by the test-time continuous uncertainty-to-weight mapping")
    parser.add_argument("--finetune_beta_v2", type=float, default=0.02, help="Beta used by finetune version v2 weighting")
    parser.add_argument("--finetune_beta_v3", type=float, default=0.02, help="Beta used by finetune version v3 weighting")
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix appended to the test output folder name for sweep runs")

    # Visibility / filtering mode
    parser.add_argument("--visibility_mode", type=str, default="pred", choices=["pred", "gt", "both"], help="How to determine which landmarks exist: 'pred' = prediction-based, 'gt' = ground truth based, 'both' = intersection")
    parser.add_argument("--pred_visibility_thresh", type=float, default=0.0, help="Probability threshold on the max predicted heatmap value for a landmark to be considered visible")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data/DeepFluoro", help="Directory containing training images")
    parser.add_argument("--csv_file", type=str, default="train_label.csv", help="CSV file containing training annotations")
    parser.add_argument("--model_weight_dir", type=str, default="model_weight", help="Directory to save model weights")
    parser.add_argument('--task_type', type=str, default='hard', choices=['easy', 'medium', 'hard'], help="Task type to process")

    # Image/label settings
    parser.add_argument("--image_resize", type=int, default=512, help="Target image size after resizing (must be divisible by 32)")
    parser.add_argument("--n_landmarks", type=int, default=14, help="Number of landmarks in total")
    parser.add_argument("--invisible_landmarks", type=str2bool, default=True, choices=[True, False], help="Whether there are invisible landmarks in the dataset")

    # Model parameters
    parser.add_argument("--encoder_depth", type=int, default=5, help="Depth of the encoder in the U-Net model")
    parser.add_argument("--decoder_channels", type=arg_as_list, default="[256, 128, 64, 32, 16]", help="List of channels in the decoder of the U-Net model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--model_weight_name", type=str, default="", help="Pretrained model weight name (if any)")

    # Training parameters
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing before training")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=350, help="Number of training epochs")
    parser.add_argument("--dilation_iters", type=int, default=65, help="Number of iterations for binary dilation")
    parser.add_argument("--erosion_freq", type=int, default=50, help="Apply erosion every N epochs")
    parser.add_argument("--erosion_iters", type=int, default=10, help="Number of iterations for binary erosion")

    # Testing parameters
    parser.add_argument("--test_prediction", action="store_true", help="Run prediction on test set")

    # Wandb parameters
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="Landmark Detection Patient-Held-Out", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="your_entity", help="Weights & Biases entity name")
    parser.add_argument("--wandb_name", type=str, default="baseline", help="Weights & Biases run name")

    # Parameters for DRR
    parser.add_argument('--sdd', type=float, default=1020.0, help='Source to Detector Distance')
    parser.add_argument('--svd', type=float, default=400.0, help='Source-to-Volume Distance (SVD) in mm')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--sample_size', type=int, default=600, help='Number of samples')

    args = parser.parse_args()

    # Fix randomness
    set_seed(args.seed)

    # Create necessary directories
    # 1. Results directories
    args.result_dir = f"results/{args.model_type}/{args.wandb_name}"
    os.makedirs(f"{args.result_dir}/visualization", exist_ok=True)
    os.makedirs(f"{args.result_dir}/graph", exist_ok=True)
    os.makedirs(f"{args.result_dir}/train_results", exist_ok=True)

    # 2. Visualization directory
    args.vis_dir = f"visualizations/{args.model_type}/{args.wandb_name}"
    os.makedirs(args.vis_dir, exist_ok=True)

    # 3. Model weight directory
    os.makedirs(f"{args.model_weight_dir}/{args.model_type}/", exist_ok=True)

    landmark_prediction_train(args)
