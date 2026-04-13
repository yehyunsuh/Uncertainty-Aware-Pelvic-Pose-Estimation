import copy
import os
import warnings

import argparse
import torch
import pandas as pd

from src.train.model import UNet
from src.train.utils import set_seed, str2bool, arg_as_list
from src.train_deepfluoro_real.data_loader import LANDMARK_NAMES
from src.train_deepfluoro_real.test import evaluate_test_set
from src.train_deepfluoro_real.train import train
from src.train_patient_held_out.log import initiate_wandb


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")


SPECIMEN_IDS = (
    "17-1882",
    "17-1905",
    "18-0725",
    "18-1109",
    "18-2799",
    "18-2800",
)


def _default_run_name(args, specimen_id: str) -> str:
    parts = [
        specimen_id,
        args.source_domain,
        args.init_mode,
        args.task_type,
    ]
    if args.output_tag:
        parts.append(args.output_tag)
    return "_".join(parts)


def _prepare_output_dirs(args) -> None:
    args.result_dir = f"results/{args.model_type}/{args.wandb_name}"
    args.vis_dir = f"visualizations/{args.model_type}/{args.wandb_name}"

    os.makedirs(f"{args.result_dir}/visualization", exist_ok=True)
    os.makedirs(f"{args.result_dir}/graph", exist_ok=True)
    os.makedirs(f"{args.result_dir}/train_results", exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(f"{args.model_weight_dir}/{args.model_type}", exist_ok=True)


def _load_initial_weights(args, model, device) -> None:
    if args.init_mode != "synthetic":
        return

    weight_path = args.synthetic_weight_path
    if not weight_path:
        weight_path = (
            f"{args.synthetic_model_weight_dir}/"
            f"{args.specimen_id}_{args.task_type}_dist.pth"
        )

    print(f"Loading synthetic initialization from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def _train_single_specimen(base_args, specimen_id: str):
    args = copy.deepcopy(base_args)
    args.specimen_id = specimen_id
    args.finetune_mode = False
    if args.wandb_name and base_args.all_specimens:
        args.wandb_name = f"{args.wandb_name}_{specimen_id}"
    elif not args.wandb_name:
        args.wandb_name = _default_run_name(args, specimen_id)

    _prepare_output_dirs(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Held-out specimen: {args.specimen_id}")

    if args.wandb:
        import wandb

        initiate_wandb(args)

    model = UNet(args, device)
    _load_initial_weights(args, model, device)

    if args.train_mode:
        train(args, model, device)
        if args.wandb:
            wandb.finish()
        return

    if args.test_mode:
        checkpoint_suffix = args.checkpoint_metric
        weight_path = (
            args.test_weight_path
            or f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_{checkpoint_suffix}.pth"
        )
        args.test_weight_path = weight_path
        print(f"Loading test checkpoint from {weight_path}")
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        summary, _, _ = evaluate_test_set(args, model, device)
        if args.wandb:
            wandb.finish()
        return summary

    print("Please specify --train_mode or --test_mode.")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patient-held-out DeepFluoro real-image training."
    )

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--specimen_id", type=str, default="17-1882", help="Held-out specimen ID")
    parser.add_argument("--all_specimens", action="store_true", help="Run training for all six held-out specimens")
    parser.add_argument("--model_type", type=str, default="train_deepfluoro_real", help="Output model group name")
    parser.add_argument("--train_mode", action="store_true", help="Run supervised training")
    parser.add_argument("--test_mode", action="store_true", help="Run held-out specimen testing")
    parser.add_argument("--preprocess", action="store_true", help="Build real-image manifests before training")
    parser.add_argument("--source_domain", type=str, default="real", choices=["real", "synthetic", "mixed"], help="Training domain")
    parser.add_argument("--init_mode", type=str, default="imagenet", choices=["imagenet", "synthetic"], help="Initialization mode")
    parser.add_argument("--real_train_ratio", type=float, default=0.8, help="Per-specimen real train split ratio")
    parser.add_argument("--rotation_prob", type=float, default=0.5, help="Probability of random 0-359 degree rotation during training")
    parser.add_argument("--task_type", type=str, default="hard", choices=["easy", "medium", "hard"], help="Synthetic DRR difficulty")
    parser.add_argument("--data_dir", type=str, default="data/DeepFluoro", help="Input data root")
    parser.add_argument("--h5_path", type=str, default="data/ipcai_2020_full_res_data.h5", help="DeepFluoro H5 with projection calibration and GT poses for pose evaluation")
    parser.add_argument("--model_weight_dir", type=str, default="model_weight", help="Directory to save new weights")
    parser.add_argument("--synthetic_model_weight_dir", type=str, default="model_weight/patient_held_out", help="Directory containing synthetic held-out checkpoints")
    parser.add_argument("--synthetic_weight_path", type=str, default="", help="Optional explicit synthetic checkpoint path")
    parser.add_argument("--synthetic_model_type", type=str, default="patient_held_out", help="Synthetic CSV source namespace")
    parser.add_argument("--image_resize", type=int, default=512, help="Target image size")
    parser.add_argument("--n_landmarks", type=int, default=len(LANDMARK_NAMES), help="Number of landmarks")
    parser.add_argument("--invisible_landmarks", type=str2bool, default=True, choices=[True, False], help="Whether invisible landmarks exist")
    parser.add_argument("--encoder_depth", type=int, default=5, help="Encoder depth")
    parser.add_argument("--decoder_channels", type=arg_as_list, default="[256, 128, 64, 32, 16]", help="Decoder channels")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--epochs", type=int, default=350, help="Training epochs")
    parser.add_argument("--dilation_iters", type=int, default=65, help="Binary dilation iterations")
    parser.add_argument("--erosion_freq", type=int, default=50, help="Apply erosion every N epochs")
    parser.add_argument("--erosion_iters", type=int, default=10, help="Number of erosion iterations")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="DeepFluoro Real Held-Out", help="Weights & Biases project")
    parser.add_argument("--wandb_entity", type=str, default="your_entity", help="Weights & Biases entity")
    parser.add_argument("--wandb_name", type=str, default="", help="Run name; auto-generated when omitted")
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix for auto-generated run names")
    parser.add_argument("--checkpoint_metric", type=str, default="dist", choices=["dist", "loss"], help="Checkpoint suffix to load during testing")
    parser.add_argument("--test_weight_path", type=str, default="", help="Optional explicit checkpoint path for testing")

    args = parser.parse_args()
    set_seed(args.seed)

    specimen_ids = SPECIMEN_IDS if args.all_specimens else (args.specimen_id,)
    all_test_summaries = []
    for specimen_id in specimen_ids:
        result = _train_single_specimen(args, specimen_id)
        if args.test_mode and result is not None:
            all_test_summaries.append(result)

    if args.test_mode and all_test_summaries:
        cohort_df = pd.DataFrame(all_test_summaries)
        cohort_dir = f"results/{args.model_type}"
        os.makedirs(cohort_dir, exist_ok=True)
        cohort_path = os.path.join(
            cohort_dir,
            f"cohort_test_summary_{args.wandb_name or 'auto'}.csv",
        )
        cohort_df.to_csv(cohort_path, index=False)
        print(f"Saved cohort test summary to {cohort_path}")
