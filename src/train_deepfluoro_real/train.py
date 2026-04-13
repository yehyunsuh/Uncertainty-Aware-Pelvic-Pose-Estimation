import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.train.visualization import overlay_pred_coords, plot_training_results
from src.train_deepfluoro_real.data_loader import dataloader
from src.train_patient_held_out.log import log_results


def train_model(model, device, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    progress = tqdm(
        train_loader,
        desc="Train",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, (images, masks, _, _) in enumerate(progress, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f"{total_loss / batch_idx:.4f}")

    return total_loss / max(len(train_loader), 1)


def evaluate_model(args, model, device, val_loader, epoch):
    model.eval()
    total_loss = 0.0
    all_pred_coords = []
    all_gt_coords = []
    all_dice = []
    coords_image = None

    with torch.no_grad():
        progress = tqdm(
            val_loader,
            desc="Val",
            leave=False,
            dynamic_ncols=True,
        )
        for idx, (images, masks, _, landmarks) in enumerate(progress):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            loss = nn.BCEWithLogitsLoss()(outputs, masks)
            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / (idx + 1):.4f}")

            probs = torch.sigmoid(outputs)
            batch_size, channels, _, width = probs.shape
            probs_flat = probs.view(batch_size, channels, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((batch_size, channels, 2), device=device)
            for b in range(batch_size):
                for c in range(channels):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, width)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            gt_coords = torch.tensor(landmarks, dtype=torch.float32, device=device)
            if gt_coords.ndim == 2:
                gt_coords = gt_coords.unsqueeze(0)

            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)

            pred_bin = (probs > 0.5).float()
            intersection = (pred_bin * masks).sum(dim=(2, 3))
            union = pred_bin.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            all_dice.append(dice)

            if idx == 0:
                coords_image = overlay_pred_coords(
                    args,
                    images,
                    pred_coords,
                    gt_coords,
                    epoch,
                    args.epochs,
                    idx,
                )

    avg_loss = total_loss / max(len(val_loader), 1)
    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)
    all_dice = torch.cat(all_dice, dim=0)

    diff = all_pred_coords - all_gt_coords
    dists = torch.norm(diff, dim=2)
    mask = (all_gt_coords != 0).any(dim=2)
    dists[~mask] = float("nan")

    evaluate_model.last_dice = all_dice
    mean_dice = all_dice.mean().item()

    return avg_loss, dists, mean_dice, None, None, coords_image


def train(args, model, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_mean_error = float("inf")
    best_val_loss = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "mean_landmark_error": [],
        "landmark_errors": {str(c): [] for c in range(args.n_landmarks)},
        "mean_dice": [],
        "dice_scores": {str(c): [] for c in range(args.n_landmarks)},
        "best_val_loss": [],
        "best_mean_error": [],
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        if epoch % args.erosion_freq == 0:
            if epoch != 0:
                args.dilation_iters = max(args.dilation_iters - args.erosion_iters, 1)

            image_size = args.image_resize ** 2
            n_dilated = 1 + 2 * args.dilation_iters * (args.dilation_iters + 1)
            weight_ratio = (
                (image_size * 100 / n_dilated)
                / (image_size * 100 / max(image_size - n_dilated, 1))
            )
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(weight_ratio).to(device)
            )

            train_loader, val_loader = dataloader(args, data_type="train", epoch=epoch)
            print(f"Loss weight: {weight_ratio:.4f}")

        train_loss = train_model(model, device, train_loader, optimizer, loss_fn)
        val_loss, dists, mean_dice, _, _, _ = evaluate_model(
            args, model, device, val_loader, epoch
        )

        mean_dist = torch.nanmean(dists).item()
        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Mean Dist: {mean_dist:.4f} | "
            f"Mean Dice: {mean_dice:.4f}"
        )

        if mean_dist < best_mean_error:
            best_mean_error = mean_dist
            torch.save(
                model.state_dict(),
                f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_dist.pth",
            )
            print("Saved new best model based on mean distance.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                f"{args.model_weight_dir}/{args.model_type}/{args.wandb_name}_loss.pth",
            )
            print("Saved new best model based on validation loss.")

        if args.wandb:
            log_results(
                train_loss,
                val_loss,
                mean_dist,
                mean_dice,
                best_mean_error,
                best_val_loss,
            )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_landmark_error"].append(mean_dist)
        history["mean_dice"].append(mean_dice)
        history["best_mean_error"].append(best_mean_error)
        history["best_val_loss"].append(best_val_loss)

        for c in range(dists.shape[1]):
            history["landmark_errors"][str(c)].append(torch.nanmean(dists[:, c]).item())
            history["dice_scores"][str(c)].append(
                evaluate_model.last_dice[:, c].mean().item()
            )

    plot_training_results(args, history, graph_dir=f"{args.result_dir}/graph")

    rows = []
    for i, epoch in enumerate(history["epoch"]):
        row = [
            epoch,
            history["train_loss"][i],
            history["val_loss"][i],
            history["mean_landmark_error"][i],
        ]
        row += [history["landmark_errors"][str(c)][i] for c in range(args.n_landmarks)]
        row.append(history["mean_dice"][i])
        row += [history["dice_scores"][str(c)][i] for c in range(args.n_landmarks)]
        row.append(history["best_val_loss"][i])
        row.append(history["best_mean_error"][i])
        rows.append(row)

    columns = ["epoch", "train_loss", "val_loss", "mean_dist"]
    columns += [f"landmark{c + 1}_dist" for c in range(args.n_landmarks)]
    columns += ["mean_dice"]
    columns += [f"landmark{c + 1}_dice" for c in range(args.n_landmarks)]
    columns += ["best_val_loss", "best_mean_error"]

    df = pd.DataFrame(rows, columns=columns)
    out_path = os.path.join(args.result_dir, "train_results", "training_log.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved training log to {out_path}")

    return model
