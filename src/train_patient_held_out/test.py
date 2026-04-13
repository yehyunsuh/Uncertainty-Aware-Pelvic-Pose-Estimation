import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.train_patient_held_out.data_loader import dataloader

from src.test.data import load_or_compute_per_image_uncertainty, save_predictions_csv
from src.test.uncertainty import uncertainty_evaluation
from src.test.result import save_csv
from src.test.visualization import plot_overall_trends
from src.sweep.main import save_run_summary


# Your specific color palette
LM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854"
]

def hex_to_bgr(hex_color):
    """Converts hex string (e.g., '#e41a1c') to BGR tuple (28, 26, 228) for OpenCV."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR

# Pre-compute BGR colors for OpenCV
LM_COLORS_BGR = [hex_to_bgr(c) for c in LM_COLORS]


def test_model(args, model, device, test_loader, overlay_dir=None):
    model.eval()

    all_preds_coords = []
    all_gt_coords = []
    all_image_names = []

    for idx, (image, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(test_loader, desc="Testing")):
        image = image.to(device)
        landmarks = landmarks.to(device)
        if landmarks.ndim == 2:
            landmarks = landmarks.unsqueeze(0)

        zero_mask = (landmarks == 0).all(dim=2)
        landmarks[zero_mask] = float('nan')
        all_gt_coords.append(landmarks)

        B, C = 1, args.n_landmarks

        # ---- single forward pass ----
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        _, _, H, W = probs.shape

        flat = probs.view(B, C, -1)
        max_idx = flat.argmax(dim=2)

        preds = torch.zeros((B, C, 2), device=device)
        for c in range(C):
            idx1d = max_idx[0, c].item()
            y, x = divmod(idx1d, W)
            preds[0, c] = torch.tensor([x, y], device=device)

        # ---------- save overlay image ----------
        if overlay_dir is not None:
            # 1. Load Original Image from Disk
            img_path = f'data/DeepFluoro/{specimen_id[0]}/drr_projections_hard/{specimen_id[0]}_{image_name[0]}'
            cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # 2. Save original (optional, as per your request)
            out_name_orig = f"{image_name[0].split('.')[0]}_orig.png"
            out_path_orig = os.path.join(overlay_dir, out_name_orig)
            os.makedirs(os.path.dirname(out_path_orig), exist_ok=True)
            cv2.imwrite(out_path_orig, cv2_img)

            # 3. Prepare image for Drawing
            img_disp = cv2_img.copy()
            if img_disp.dtype == np.uint16:
                img_disp = (img_disp / 256).astype(np.uint8)
            if img_disp.ndim == 2:
                img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

            gt_np = landmarks[0].detach().cpu().numpy()      # [C, 2]
            pred_np = preds[0].detach().cpu().numpy()        # [C, 2]

            gt_vis   = ~np.isnan(gt_np).any(axis=1)
            pred_vis = ~np.isnan(pred_np).any(axis=1)

            for c in range(C):
                # Use modulo to cycle colors if C > len(LM_COLORS)
                color_idx = c % len(LM_COLORS_BGR)
                
                # Ground Truth in BLUE
                if gt_vis[c]:
                    xg, yg = gt_np[c]
                    cv2.circle(img_disp, (int(round(xg)), int(round(yg))), 8, (255, 0, 0), -1)
                
                # # Prediction in SPECIFIC COLOR
                # if pred_vis[c]:
                #     xp, yp = pred_np[c]
                #     # cv2.circle(img_disp, (int(round(xp)), int(round(yp))), 4, LM_COLORS_BGR[color_idx], -1)
                #     cv2.circle(img_disp, (int(round(xp)), int(round(yp))), 8, (0, 0, 255), -1)

            out_name = image_name[0]
            out_path = os.path.join(overlay_dir, out_name)
            cv2.imwrite(out_path, img_disp)

        # shape [S=1, B, C, 2]
        sim_coords_batch = preds.unsqueeze(0)
        all_preds_coords.append(sim_coords_batch)
        all_image_names.append(image_name[0])

    all_preds_coords = torch.cat(all_preds_coords, dim=1)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    return all_preds_coords, all_gt_coords, all_image_names


def test_model_uncertainty(args, model, device, test_loader, overlay_dir=None):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()

    all_sim_coords = []
    all_gt_coords = []
    all_image_names = []

    for idx, (image, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(test_loader, desc="Testing Uncertainty")):
        image = image.to(device)
        landmarks = landmarks.to(device)
        if landmarks.ndim == 2: 
            landmarks = landmarks.unsqueeze(0)

        zero_mask = (landmarks == 0).all(dim=2)
        landmarks[zero_mask] = float('nan')
        all_gt_coords.append(landmarks)

        B, C = 1, args.n_landmarks
        sim_coords_batch = []

        # ---- Run Simulations ----
        for _ in range(args.n_simulations):
            outputs = model(image)
            probs = torch.sigmoid(outputs)
            _, _, H, W = probs.shape

            flat = probs.view(B, C, -1)
            max_idx = flat.argmax(dim=2)

            preds = torch.zeros((B, C, 2), device=device)
            for c in range(C):
                idx1d = max_idx[0, c].item()
                y, x = divmod(idx1d, W)
                preds[0, c] = torch.tensor([x, y], device=device)
            sim_coords_batch.append(preds.clone())

        # [S, B, C, 2]
        sim_tensor = torch.stack(sim_coords_batch)
        all_sim_coords.append(sim_tensor)
        all_image_names.append(image_name[0])

        # ---------- save uncertainty overlay image (Matplotlib style) ----------
        if overlay_dir is not None:
            # 1. Load Original Image path
            img_path = f'data/DeepFluoro/{specimen_id[0]}/drr_projections_hard/{specimen_id[0]}_{image_name[0]}'
            
            # Setup output path
            out_path = os.path.join(overlay_dir, image_name[0].replace('.png', '_uncertainty.png'))
            
            # 2. Visualization Logic (Inverted + Matplotlib Scatter)
            # Load with OpenCV to handle formats, then convert to RGB
            cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # Normalize for display if 16-bit
            if cv2_img.dtype == np.uint16:
                cv2_img = (cv2_img / 256).astype(np.uint8)
            
            # Ensure RGB for Matplotlib
            if cv2_img.ndim == 2:
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            # Create Plot
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(img_rgb)
            ax.set_axis_off()

            # Iterate over simulations and draw
            # sim_tensor is [S, 1, C, 2] -> we need [S, C, 2]
            mc_coords = sim_tensor[:, 0, :, :].cpu().numpy() # [S, C, 2]

            for c in range(C):
                color_hex = LM_COLORS[c % len(LM_COLORS)]
                
                # Get all simulation points for landmark C
                # shape [S, 2]
                coords_c = mc_coords[:, c, :]
                
                # Filter out NaNs if any (though usually predictions aren't NaN)
                valid_mask = ~np.isnan(coords_c).any(axis=1)
                coords_c = coords_c[valid_mask]

                if len(coords_c) > 0:
                    ax.scatter(coords_c[:, 0], coords_c[:, 1], 
                               s=200,               # size
                               color=color_hex,    # specific color
                               alpha=0.7)          # transparency

            plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    all_sim_coords = torch.cat(all_sim_coords, dim=1)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    return all_sim_coords, all_gt_coords, all_image_names


def test_model_uncertainty_parallel(args, model, device, test_loader, overlay_dir=None, sim_batch_size=40):
    """
    Args:
        sim_batch_size (int): Number of simulations to run in parallel on the GPU. 
                              Default is 40 (fit for ~46GB GPU).
    """
    model.eval()
    
    # Enable dropout during inference for MC sampling
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    all_sim_coords = []
    all_gt_coords = []
    all_image_names = []

    for idx, (image, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(test_loader, desc="Testing Uncertainty")):
        image = image.to(device)
        landmarks = landmarks.to(device)
        
        # --- Handle Ground Truth ---
        if landmarks.ndim == 2: 
            landmarks = landmarks.unsqueeze(0)

        zero_mask = (landmarks == 0).all(dim=2)
        landmarks[zero_mask] = float('nan')
        all_gt_coords.append(landmarks)
        
        # --- Parallel Simulation ---
        total_sims = args.n_simulations
        sim_coords_list = []
        
        # Process in chunks (batches) to manage GPU memory
        # e.g., if n_simulations=100 and sim_batch_size=40, loops: 40, 40, 20
        for i in range(0, total_sims, sim_batch_size):
            # 1. Determine batch size for this iteration
            current_batch_size = min(sim_batch_size, total_sims - i)
            
            # 2. Expand single image to batch: [1, 3, H, W] -> [Batch, 3, H, W]
            image_batch = image.repeat(current_batch_size, 1, 1, 1)
            
            # 3. Forward Pass (No grad needed for inference, saves massive memory)
            with torch.no_grad():
                outputs = model(image_batch)
                probs = torch.sigmoid(outputs)  # [Batch, C, H, W]
            
            B_curr, C, H, W = probs.shape
            
            # 4. Vectorized Coordinate Extraction
            # Flatten spatial dims: [Batch, C, H*W]
            flat = probs.view(B_curr, C, -1)
            max_idx = flat.argmax(dim=2)  # [Batch, C]
            
            # Convert 1D indices to 2D (x, y) coordinates
            y_coords = torch.div(max_idx, W, rounding_mode='floor')
            x_coords = max_idx % W
            
            # Stack to create [Batch, C, 2]
            preds_batch = torch.stack([x_coords, y_coords], dim=2).float()
            sim_coords_list.append(preds_batch)

        # Combine chunks: [Total_Sims, C, 2]
        sim_tensor = torch.cat(sim_coords_list, dim=0)
        
        # Reshape to match original format [Total_Sims, 1 (Image Batch), C, 2]
        sim_tensor = sim_tensor.unsqueeze(1)
        
        all_sim_coords.append(sim_tensor)
        all_image_names.append(image_name[0])

        # ---------- Save Uncertainty Overlay ----------
        if overlay_dir is not None:
            # Note: This logic assumes batch_size=1 for the data loader
            img_path = f'data/DeepFluoro/{specimen_id[0]}/drr_projections_hard/{specimen_id[0]}_{image_name[0]}'
            out_path = os.path.join(overlay_dir, image_name[0].replace('.png', '_uncertainty_parallel.png'))
            
            cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if cv2_img.dtype == np.uint16:
                cv2_img = (cv2_img / 256).astype(np.uint8)
            
            if cv2_img.ndim == 2:
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(img_rgb)
            ax.set_axis_off()

            # Plotting: sim_tensor is [S, 1, C, 2]
            mc_coords = sim_tensor[:, 0, :, :].cpu().numpy() # [S, C, 2]

            for c in range(args.n_landmarks):
                color_hex = LM_COLORS[c % len(LM_COLORS)]
                coords_c = mc_coords[:, c, :]
                
                valid_mask = ~np.isnan(coords_c).any(axis=1)
                coords_c = coords_c[valid_mask]

                if len(coords_c) > 0:
                    ax.scatter(coords_c[:, 0], coords_c[:, 1], 
                               s=200, color=color_hex, alpha=0.7)

            plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    all_sim_coords = torch.cat(all_sim_coords, dim=1)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    return all_sim_coords, all_gt_coords, all_image_names


def test(args, model, model_dropout, device):
    folder_name = f"prediction_{args.dropout_rate}_{args.model_weight_name}"
    if args.output_tag:
        folder_name = f"{folder_name}_{args.output_tag}"
    args.save_folder_name = folder_name
    csv_dir = os.path.join(args.vis_dir, args.save_folder_name, "csv_results")
    plot_dir = os.path.join(args.vis_dir, args.save_folder_name, "uncertainty_plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    test_loader = dataloader(args, data_type='test')
    if args.test_prediction:
        # start_time = time.time()
        # preds, gt_coords, image_names = test_model(args, model, device, test_loader, overlay_dir=None)
        # end_time = time.time()
        # print(f"Test Prediction Time: {end_time - start_time:.2f} seconds")
        preds, gt_coords, image_names = test_model(args, model, device, test_loader, overlay_dir=f'{args.vis_dir}/{args.save_folder_name}/overlays')
        save_predictions_csv(args, image_names, csv_dir, preds, gt_coords, prefix="predictions")

        start_time = time.time()
        mc_preds, _, _ = test_model_uncertainty(args, model_dropout, device, test_loader, overlay_dir=None)
        end_time = time.time()
        print(f"Test Uncertainty Time: {end_time - start_time:.2f} seconds")
        # mc_preds, _, _ = test_model_uncertainty(args, model_dropout, device, test_loader, overlay_dir=f'{args.vis_dir}/{args.save_folder_name}/overlays')
        # save_predictions_csv(args, image_names, csv_dir, mc_preds, gt_coords, prefix="mc_predictions")

        start_time = time.time()
        
        mc_preds, _, _ = test_model_uncertainty_parallel(
            args, 
            model_dropout, 
            device, 
            test_loader, 
            overlay_dir=f'{args.vis_dir}/{args.save_folder_name}/overlays',
            sim_batch_size=100
        )
        
        end_time = time.time()
        print(f"Test Uncertainty Time: {end_time - start_time:.2f} seconds")
        save_predictions_csv(args, image_names, csv_dir, mc_preds, gt_coords, prefix="mc_predictions")

    df_unc, cluster_pivot = load_or_compute_per_image_uncertainty(csv_dir, dropout_rate=args.dropout_rate)

    suffix = f"perImage_{args.visibility_mode}"
    start_time = time.time()
    all_results = uncertainty_evaluation(args, model, test_loader, device, cluster_pivot)
    end_time = time.time()
    print(f"Uncertainty Evaluation Time: {end_time - start_time:.2f} seconds")
    
    results_df = save_csv(args, all_results, suffix=suffix)
    plot_overall_trends(args, results_df, plot_dir, suffix=suffix)
    save_run_summary(args, results_df, suffix=suffix)
