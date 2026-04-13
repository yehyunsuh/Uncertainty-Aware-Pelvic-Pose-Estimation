import os
import math
import torch
import wandb
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from glob import glob

from src.train_patient_held_out.data_loader import dataloader
from src.train_patient_held_out.log import log_results
from src.train.visualization import (
    overlay_gt_masks, overlay_pred_masks, overlay_pred_coords,
    create_gif, plot_training_results
)
from src.test.pose_estimation import pose_estimation, pose_estimation_differentiable, pose_estimation_weighted_differentiable


def soft_argmax_2d(probs, temperature=1e-8):
    """
    Differentiable soft-argmax for heatmaps/probability maps.

    Args:
        probs: tensor of shape [B, C, H, W]
        temperature: temperature for softmax (lower = harder argmax)

    Returns:
        coords: tensor of shape [B, C, 2] → (x, y) coordinates
    """

    B, C, H, W = probs.shape
    device = probs.device

    # [B, C, H*W]
    probs_flat = probs.view(B, C, -1)

    # Softmax over spatial dimension
    softmax_probs = torch.softmax(probs_flat / temperature, dim=2)

    # Build coordinate grid (H*W)
    rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1)  # [H*W]
    cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1)  # [H*W]

    # reshape to [1,1,H*W] for broadcasting
    rows_b = rows.view(1, 1, -1)
    cols_b = cols.view(1, 1, -1)

    # Weighted sum → expected coordinate
    soft_y = torch.sum(rows_b * softmax_probs, dim=2)  # [B,C]
    soft_x = torch.sum(cols_b * softmax_probs, dim=2)  # [B,C]

    # Stack as (x, y)
    coords = torch.stack([soft_x, soft_y], dim=2)  # [B,C,2]

    return coords


def compute_euler_error_wrapped(pred_deg, gt_deg):
    """
    Computes the shortest difference between angles in degrees.
    Handles wrapping: 359 vs 1 becomes 2.
    Also handles unbounded inputs: 721 vs 0 becomes 1.
    """
    # Calculate raw difference
    diff = pred_deg - gt_deg
    
    # Wrap to [-180, 180]
    # (diff + 180) % 360 - 180 ensures the result is always the shortest path
    wrapped_diff = (diff + 180) % 360 - 180
    
    return np.abs(wrapped_diff)


def differentiable_wrapped_mse(pred, target):
    """
    Computes MSE loss respecting angular wrapping [-180, 180].
    Assumes inputs are in degrees.
    """
    diff = pred - target
    # Wrap to [-180, 180]
    # In PyTorch, fmod can handle modulo, but we need the specific wrapping logic
    # wrapped = (diff + 180) % 360 - 180
    
    # Differentiable approximate logic or just fix the huge outliers:
    # A simple way for rotation loss in degrees is:
    wrapped_diff = diff - 360 * torch.round(diff / 360)
    
    return torch.sqrt(torch.mean(wrapped_diff ** 2))


def train_finetune_model(args, model, model_uncertainty, device, train_loader, val_loader, optimizer):
    model.train()
    model_uncertainty.train()

    sdd = args.sdd
    svd = args.svd
    vdd = sdd - svd
    manual_translations_list = torch.tensor([[0.0, svd, 0.0]])

    best_dists_filtered = np.inf

    batch_log_rmse_rot_gt_input = 0
    batch_log_rmse_trans_gt_input = 0
    batch_log_rmse_rot_w_gt_input = 0
    batch_log_rmse_trans_w_gt_input = 0
    batch_log_rmse_rot_pred = 0
    batch_log_rmse_trans_pred = 0
    batch_log_rmse_rot_w_pred = 0
    batch_log_rmse_trans_w_pred = 0
    batch_rmse_error = 0
    batch_mean_pixel_error = 0
    batch_rot_loss = 0
    batch_trans_loss = 0
    batch_bce_loss = 0
    batch_loss = 0
    for batch_idx, (images, masks, specimen_id, image_name, landmarks, pose_params) in tqdm(enumerate(train_loader), desc="Finetune Training"):
        if batch_idx % 10 == 0:
            dists, dists_filtered, val_rot_err, val_trans_err, val_loss = evaluate_finetune_model(args, model, device, val_loader, sdd, svd, vdd, manual_translations_list)
            print(f"==== Batch {batch_idx}: Validation Mean Dist: {torch.nanmean(dists).item():.4f} | {torch.nanmean(dists_filtered).item():.4f}, Val Rot Error: {val_rot_err:.4f}, Val Trans Error: {val_trans_err:.4f}, Loss: {val_loss:.4f} ====")

            # save model weight
            torch.save(model.state_dict(), f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_dist_finetune.pth")
            # torch.save(model.state_dict(), f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_loss.pth")

            model.train()

            if batch_idx > 0 and args.wandb:
                wandb.log({
                    "Train Rot Loss": batch_rot_loss / 10,
                    "Train Trans Loss": batch_trans_loss / 10,
                    "Train Total Loss": batch_loss / 10,
                    "Train RMSE Error": batch_rmse_error / 10,
                    "Train Mean pixel Error": batch_mean_pixel_error / 10,
                    "Train BCE Loss": batch_bce_loss /10,

                    "Val Mean Landmark Error": torch.nanmean(dists).item(),
                    "Val Mean Landmark Error Filtered": torch.nanmean(dists_filtered).item(),
                    "Val Rotation Error": val_rot_err,
                    "Val Translation Error": val_trans_err,
                    "Val BCE Loss": val_loss,

                    "Train GT Rot RMSE (Unweighted)": batch_log_rmse_rot_gt_input / 10,
                    "Train GT Trans RMSE (Unweighted)": batch_log_rmse_trans_gt_input / 10,
                    "Train GT Rot RMSE (Weighted)": batch_log_rmse_rot_w_gt_input / 10,
                    "Train GT Trans RMSE (Weighted)": batch_log_rmse_trans_w_gt_input / 10,
                    
                    "Train Pred Rot RMSE (Unweighted)": batch_log_rmse_rot_pred / 10,
                    "Train Pred Trans RMSE (Unweighted)": batch_log_rmse_trans_pred / 10,
                    "Train Pred Rot RMSE (Weighted)": batch_log_rmse_rot_w_pred / 10,
                    "Train Pred Trans RMSE (Weighted)": batch_log_rmse_trans_w_pred / 10,
                })

            print(f"\n[Finetune Train] Batch {batch_idx} Averages:")
            print(f"  Rot Loss: {batch_rot_loss / 10:.4f}, Trans Loss: {batch_trans_loss / 10:.4f}, Total Loss: {batch_loss / 10:.4f}")
            print(f"  RMSE Error: {batch_rmse_error / 10:.4f} pixels, Mean Pixel Error: {batch_mean_pixel_error / 10:.4f} pixels")

            batch_log_rmse_rot_gt_input = 0
            batch_log_rmse_trans_gt_input = 0
            batch_log_rmse_rot_w_gt_input = 0
            batch_log_rmse_trans_w_gt_input = 0
            batch_log_rmse_rot_pred = 0
            batch_log_rmse_trans_pred = 0
            batch_log_rmse_rot_w_pred = 0
            batch_log_rmse_trans_w_pred = 0
            batch_rmse_error = 0
            batch_mean_pixel_error = 0
            batch_rot_loss = 0
            batch_trans_loss = 0
            batch_bce_loss = 0
            batch_loss = 0

            model.train()

        optimizer.zero_grad()
        images = images.to(device)
        masks = masks.to(device)
        B, C, H, W = images.shape

        rotation_gt = np.array([
            np.rad2deg(pose_params[0].cpu().numpy()[0]),
            np.rad2deg(pose_params[1].cpu().numpy()[0]),
            np.rad2deg(pose_params[2].cpu().numpy()[0])
        ])
        translation_gt = np.array([
            pose_params[3].cpu().numpy()[0],
            pose_params[4].cpu().numpy()[0],
            pose_params[5].cpu().numpy()[0]
        ])

        translation_gt_adj = translation_gt.copy()
        translation_gt_adj[1] -= manual_translations_list[0, 1].item()

        landmarks = landmarks.squeeze(0)   # shape: (C, 2)
        L_Proj_gt = np.array([
            [np.nan, np.nan] if (coord[0].item() == 0 and coord[1].item() == 0)
            else [coord[0].item(), coord[1].item()]
            for coord in landmarks
        ], dtype=np.float32)
                    
        gt_visible_mask = ~np.isnan(L_Proj_gt).any(axis=1)  # [C]
        L_Proj_gt_cp = L_Proj_gt.copy()
        L_Proj_gt_cp[gt_visible_mask, 1] -= args.image_resize / 2
        L_Proj_gt_cp[gt_visible_mask, 0] -= args.image_resize / 2
        L_Proj_gt_cp[gt_visible_mask, 1] *= -1

        volume_path = f"{args.data_dir}/{specimen_id[0]}/{specimen_id[0]}_CT.nii.gz"
        volume = nib.load(volume_path)
        volume_shape = volume.shape
        center = np.array(volume_shape) // 2
        spacing = volume.header.get_zooms()[:3]
        spacing_x, spacing_y, spacing_z = spacing

        Slicer_3D_landmark = np.load(f"data/DeepFluoro/{specimen_id[0]}/{specimen_id[0]}_Landmarks_3D.npy").astype(np.float64)
        Slicer_3D_landmark -= center
        Slicer_3D_landmark[:, 0] *= spacing_x
        Slicer_3D_landmark[:, 1] *= spacing_y
        Slicer_3D_landmark[:, 2] *= spacing_z
        Slicer_3D_landmark += center

        Point_3D_landmark = (Slicer_3D_landmark - center).astype(np.float32)
        Point_3D_landmark[:, 1] += ((sdd - svd) / spacing_y)
        Point_3D_landmark[:, 2] -= (manual_translations_list[0, 2].item() / spacing_z)

        outputs = model(images)
        pred_coords = soft_argmax_2d(torch.sigmoid(outputs))  # [B, C, 2]
        pred_coords_transformed = pred_coords.clone()
        pred_coords_transformed[..., 0] -= args.image_resize / 2
        pred_coords_transformed[..., 1] -= args.image_resize / 2
        pred_coords_transformed[..., 1] *= -1
        L_Proj_pred_input = pred_coords_transformed[0]
    
        with torch.no_grad():
            mc_pred_coords_all = torch.zeros(
                args.n_simulations, B, 14, 2,
                device=device,
                dtype=torch.float32
            )
            for i in range(args.n_simulations):
                outputs_unc = model_uncertainty(images)
                probs = torch.sigmoid(outputs_unc)
                B, C, H, W = probs.shape
                probs_flat = probs.view(B, C, -1)
                soft_coords = soft_argmax_2d(torch.sigmoid(outputs_unc))
                mc_pred_coords_all[i] = soft_coords

        # Calculate Uncertainty
        # mc_pred_coords_all = mc_pred_coords_all.requires_grad_()
        uncertainty_var = mc_pred_coords_all.var(dim=0, unbiased=False)   # [B,14,2]
        uncertainty_rms = torch.sqrt(uncertainty_var.sum(dim=2))       # [B,14]
        
        # Version 1 — Inverse uncertainty_rms (normalized)
        eps = 1e-6
        inv = 1.0 / (uncertainty_rms + eps)                # [B, 14]
        inv_norm = inv / inv.max(dim=1, keepdim=True).values
        weights_ver1 = inv_norm                      # [B, 14]

        # Version 2 — Softmax weighting
        beta2 = args.finetune_beta_v2
        soft_vals = torch.exp(-beta2 * uncertainty_rms)    # [B, 14]
        soft_norm = soft_vals / soft_vals.max(dim=1, keepdim=True).values
        weights_ver2 = soft_norm                     # [B, 14]

        # Version 3 — Soft-rank via softmax(-β · uncertainty_rms)
        beta3 = args.finetune_beta_v3
        weights_ver3 = torch.softmax(-beta3 * uncertainty_rms, dim=1)   # [B, 14]

        if args.finetune_version == "v1":
            weights = weights_ver1
        elif args.finetune_version == "v2":
            weights = weights_ver2
        elif args.finetune_version == "v3":
            weights = weights_ver3

        # Load & prepare 3D landmarks for pose estimation
        rot_torch_diff_gt, trans_torch_diff_gt = pose_estimation_differentiable(
            torch.from_numpy(Point_3D_landmark).float().to(device),
            torch.from_numpy(L_Proj_gt_cp).float().to(device),
            sdd, svd, vdd, manual_translations_list.to(device),
        )
        # print("Torch Diff Predicted Rotation:", rot_torch_diff_gt.detach().cpu().numpy())
        # print("Torch Diff Predicted Translation:", trans_torch_diff_gt.detach().cpu().numpy())

        rot_torch_diff_pred, trans_torch_diff_pred = pose_estimation_differentiable(
            torch.from_numpy(Point_3D_landmark).float().to(device),
            L_Proj_pred_input,
            sdd, svd, vdd, manual_translations_list.to(device),
        )
        # print("Torch Diff Predicted Rotation (Pred):", rot_torch_diff_pred.detach().cpu().numpy())
        # print("Torch Diff Predicted Translation (Pred):", trans_torch_diff_pred.detach().cpu().numpy())

        rot_w_torch_diff_gt, trans_w_torch_diff_gt = pose_estimation_weighted_differentiable(
            torch.from_numpy(Point_3D_landmark).float().to(device),
            torch.from_numpy(L_Proj_gt_cp).float().to(device),
            weights.squeeze(0).float().to(device),
            sdd, svd, vdd,
            manual_translations_list.to(device),
        )
        # print("Torch Weighted Diff Predicted Rotation:", rot_w_torch_diff_gt.detach().cpu().numpy())
        # print("Torch Weighted Diff Predicted Translation:", trans_w_torch_diff_gt.detach().cpu().numpy())

        rot_w_torch_diff_pred, trans_w_torch_diff_pred = pose_estimation_weighted_differentiable(
            torch.from_numpy(Point_3D_landmark).float().to(device),
            L_Proj_pred_input,
            weights.squeeze(0).float().to(device),
            sdd, svd, vdd,
            manual_translations_list.to(device),
        )
        # print("Torch Weighted Diff Predicted Rotation (Pred):", rot_w_torch_diff_pred.detach().cpu().numpy())
        # print("Torch Weighted Diff Predicted Translation (Pred):", trans_w_torch_diff_pred.detach().cpu().numpy())
        # print("Ground Truth Rotation:", rotation_gt)
        # print("Ground Truth Translation:", translation_gt_adj)

        # 1. PREPARE GROUND TRUTH TENSORS
        # Convert numpy GT to tensor for calculation
        # Shape: [3]
        gt_rot_tensor = torch.from_numpy(rotation_gt).float().to(device)
        gt_trans_tensor = torch.from_numpy(translation_gt_adj).float().to(device)

        def get_rmse(pred, target):
            return torch.sqrt(torch.mean((pred - target) ** 2))
        
        # 2. LOGGING: BASELINE ERROR (Solver Performance on GT Landmarks)
        # Standard Solver (GT Inputs)
        # log_rmse_rot_gt_input = get_rmse(rot_torch_diff_gt.detach(), gt_rot_tensor)
        per_axis_errors = compute_euler_error_wrapped(rot_torch_diff_gt.detach().cpu().numpy(), rotation_gt)
        log_rmse_rot_gt_input = torch.tensor(np.sqrt(np.mean(per_axis_errors ** 2)), device=device)
        log_rmse_trans_gt_input = get_rmse(trans_torch_diff_gt.detach(), gt_trans_tensor)
        batch_log_rmse_rot_gt_input += log_rmse_rot_gt_input.item()
        batch_log_rmse_trans_gt_input += log_rmse_trans_gt_input.item()

        # Weighted Solver (GT Inputs)
        # log_rmse_rot_w_gt_input = get_rmse(rot_w_torch_diff_gt.detach(), gt_rot_tensor)
        per_axis_errors = compute_euler_error_wrapped(rot_w_torch_diff_gt.detach().cpu().numpy(), rotation_gt)
        log_rmse_rot_w_gt_input = torch.tensor(np.sqrt(np.mean(per_axis_errors ** 2)), device=device)
        log_rmse_trans_w_gt_input = get_rmse(trans_w_torch_diff_gt.detach(), gt_trans_tensor)
        batch_log_rmse_rot_w_gt_input += log_rmse_rot_w_gt_input.item()
        batch_log_rmse_trans_w_gt_input += log_rmse_trans_w_gt_input.item()

        # print(f"\n[Baseline] Solver Error (Unweighted): Rot={log_rmse_rot_gt_input.item():.4f}, Trans={log_rmse_trans_gt_input.item():.4f}")
        # print(f"[Baseline] Solver Error (Weighted):   Rot={log_rmse_rot_w_gt_input.item():.4f}, Trans={log_rmse_trans_w_gt_input.item():.4f}")

        # 3. LOGGING: PREDICTION ERROR (Model Performance)
        # Standard Solver (Pred Inputs)
        # log_rmse_rot_pred = get_rmse(rot_torch_diff_pred.detach(), gt_rot_tensor)
        per_axis_errors = compute_euler_error_wrapped(rot_torch_diff_pred.detach().cpu().numpy(), rotation_gt)
        log_rmse_rot_pred = torch.tensor(np.sqrt(np.mean(per_axis_errors ** 2)), device=device)
        log_rmse_trans_pred = get_rmse(trans_torch_diff_pred.detach(), gt_trans_tensor)
        batch_log_rmse_rot_pred += log_rmse_rot_pred.item()
        batch_log_rmse_trans_pred += log_rmse_trans_pred.item()

        # Weighted Solver (Pred Inputs)
        # log_rmse_rot_w_pred = get_rmse(rot_w_torch_diff_pred.detach(), gt_rot_tensor)
        per_axis_errors = compute_euler_error_wrapped(rot_w_torch_diff_pred.detach().cpu().numpy(), rotation_gt)
        log_rmse_rot_w_pred = torch.tensor(np.sqrt(np.mean(per_axis_errors ** 2)), device=device)
        log_rmse_trans_w_pred = get_rmse(trans_w_torch_diff_pred.detach(), gt_trans_tensor)
        batch_log_rmse_rot_w_pred += log_rmse_rot_w_pred.item()
        batch_log_rmse_trans_w_pred += log_rmse_trans_w_pred.item()

        # print(f"\n[Prediction] Model Error (Unweighted): Rot={log_rmse_rot_pred.item():.4f}, Trans={log_rmse_trans_pred.item():.4f}")
        # print(f"[Prediction] Model Error (Weighted):   Rot={log_rmse_rot_w_pred.item():.4f}, Trans={log_rmse_trans_w_pred.item():.4f}")

        # 4. TRAINING LOSS CALCULATION (Backpropagation)
        # loss_rot = torch.sqrt(nn.MSELoss()(rot_w_torch_diff_pred, gt_rot_tensor))
        per_axis_errors = differentiable_wrapped_mse(rot_w_torch_diff_pred, gt_rot_tensor)
        loss_rot = torch.sqrt(torch.mean(per_axis_errors ** 2))
        loss_trans = torch.sqrt(nn.MSELoss()(trans_w_torch_diff_pred, gt_trans_tensor))
        loss_map = nn.BCEWithLogitsLoss()(outputs, masks)

        num_batches = len(train_loader)
        progress = batch_idx / max(1, num_batches - 1)   # in [0, 1]

        # Linear decay: start_w → end_w over the epoch
        # w_map = start_w + (end_w - start_w) * progress
        w_map = 1.0

        start_w_pose = 0.00001
        end_w_pose   = 0.01
        w_pose = start_w_pose + (end_w_pose - start_w_pose) * progress

        # Cosine decay:
        start_w = 100.0
        end_w   = 1.0
        w_map = end_w + 0.5 * (start_w - end_w) * (1 + math.cos(math.pi * progress))

        loss = (w_map * loss_map) + w_pose * (loss_rot + loss_trans)
        loss.backward()

        optimizer.step()

        batch_rot_loss += loss_rot.item()
        batch_trans_loss += loss_trans.item()
        batch_bce_loss += loss_map.item()
        batch_loss += loss.item()

        gt_landmarks_tensor = torch.from_numpy(L_Proj_gt_cp).float().to(device)
        valid_mask = torch.isfinite(gt_landmarks_tensor).all(dim=1)
        gt_valid_2d = gt_landmarks_tensor[valid_mask]
        pred_valid_2d = L_Proj_pred_input[valid_mask]
        rmse_error = torch.sqrt(nn.MSELoss()(pred_valid_2d, gt_valid_2d))
        euclidean_error = torch.sqrt(torch.sum((pred_valid_2d - gt_valid_2d) ** 2, dim=1))
        mean_pixel_error = euclidean_error.mean()

        batch_rmse_error += rmse_error.item()
        batch_mean_pixel_error += mean_pixel_error.item()

        # print(f"\n[Landmarks] 2D Projection RMSE: {rmse_error.item():.4f} pixels")
        # print(f"[Landmarks] Mean Euclidean Dist: {mean_pixel_error.item():.4f} pixels")

        model_uncertainty.load_state_dict(model.state_dict())
        model_uncertainty.train()


def evaluate_finetune_model(args, model, device, val_loader, sdd, svd, vdd, manual_translations_list):
    model.eval()
    all_pred_coords = []
    all_gt_coords = []

    total_rot_error = 0
    total_trans_error = 0
    total_val_loss = 0
    with torch.no_grad():
        # for idx, (images, specimen_id, image_name, landmarks, pose_params) in enumerate(tqdm(val_loader, desc="Validation")):
        for idx, (images, masks, specimen_id, image_name, landmarks, pose_params) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            probs = torch.sigmoid(outputs)
            val_loss = nn.BCEWithLogitsLoss()(outputs, masks)
            total_val_loss += val_loss.item()

            B, C, H, W = probs.shape
            probs_flat = probs.view(B, C, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((B, C, 2), device=device)
            for b in range(B):
                for c in range(C):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, W)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            gt_coords = landmarks.to(device).float()  # [B, C, 2]
            if gt_coords.ndim == 2:
                gt_coords = gt_coords.unsqueeze(0)

            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)

            landmarks = landmarks.squeeze(0)   # shape: (C, 2)
            L_Proj_gt = np.array([
                [np.nan, np.nan] if (coord[0].item() == 0 and coord[1].item() == 0)
                else [coord[0].item(), coord[1].item()]
                for coord in landmarks
            ], dtype=np.float32)
            base_mask = ~np.isnan(L_Proj_gt).any(axis=1)  # [C]

            # skip if visibile landmarks < 3
            if base_mask.sum() < 3:
                continue

            pred_coords_np = pred_coords[0].cpu().numpy().copy()
            pred_all = pred_coords_np.copy()

            L_Proj_pred_all = pred_all.copy()
            L_Proj_pred_all[:, 1] -= args.image_resize / 2
            L_Proj_pred_all[:, 0] -= args.image_resize / 2
            L_Proj_pred_all[:, 1] *= -1

            volume_path = f"{args.data_dir}/{specimen_id[0]}/{specimen_id[0]}_CT.nii.gz"
            volume = nib.load(volume_path)
            volume_shape = volume.shape
            center = np.array(volume_shape) // 2
            spacing = volume.header.get_zooms()[:3]
            spacing_x, spacing_y, spacing_z = spacing

            Slicer_3D_landmark = np.load(f"data/DeepFluoro/{specimen_id[0]}/{specimen_id[0]}_Landmarks_3D.npy").astype(np.float64)
            Slicer_3D_landmark -= center
            Slicer_3D_landmark[:, 0] *= spacing_x
            Slicer_3D_landmark[:, 1] *= spacing_y
            Slicer_3D_landmark[:, 2] *= spacing_z
            Slicer_3D_landmark += center

            Point_3D_landmark = (Slicer_3D_landmark - center).astype(np.float32)
            Point_3D_landmark[:, 1] += ((sdd - svd) / spacing_y)
            Point_3D_landmark[:, 2] -= (manual_translations_list[0, 2].item() / spacing_z)

            rot_all, trans_all = pose_estimation(Point_3D_landmark, L_Proj_pred_all, sdd, svd, vdd, manual_translations_list)
            # print("Predicted Rotation:", rot_all)
            # print("Predicted Translation:", trans_all)
            rotation_gt = np.array([
                np.rad2deg(pose_params[0].cpu().numpy()[0]),
                np.rad2deg(pose_params[1].cpu().numpy()[0]),
                np.rad2deg(pose_params[2].cpu().numpy()[0])
            ])
            translation_gt = np.array([
                pose_params[3].cpu().numpy()[0],
                pose_params[4].cpu().numpy()[0],
                pose_params[5].cpu().numpy()[0]
            ])
            translation_gt_adj = translation_gt.copy()
            translation_gt_adj[1] -= manual_translations_list[0, 1].item()
            # print("Ground Truth Rotation:", rotation_gt)
            # print("Ground Truth Translation:", translation_gt_adj)

            # rot_error = np.sqrt(np.mean((rot_all - rotation_gt) ** 2))
            rot_error = np.sqrt(np.mean(((rot_all - rotation_gt + 180) % 360 - 180) ** 2))
            trans_error = np.sqrt(np.mean((trans_all - translation_gt_adj) ** 2))
            
            total_rot_error += rot_error
            total_trans_error += trans_error

            if idx == 30:
                break

    all_pred_coords = torch.cat(all_pred_coords, dim=0)  # [N, C, 2]
    all_gt_coords = torch.cat(all_gt_coords, dim=0)

    # Mask (0, 0) GT for distance calculation only
    diff = all_pred_coords - all_gt_coords
    dists = torch.norm(diff, dim=2)

    dists_filtered = dists.clone()
    mask = (all_gt_coords != 0).any(dim=2)  # [B, C]
    dists_filtered[~mask] = float("nan")  # Don't include in distance average

    return dists, dists_filtered, total_rot_error / 30, total_trans_error / 30, total_val_loss / 30


def finetune(args, model, model_dropout, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    finetune_train_loader, finetune_val_loader = dataloader(args, data_type='finetune')
    train_finetune_model(args, model, model_dropout, device, finetune_train_loader, finetune_val_loader, optimizer)
