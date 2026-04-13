import os
import pandas as pd


def save_csv(args, all_results, suffix=""):
    # ============================================================
    # SAVE RESULTS TO CSV (same columns as before)
    # ============================================================
    # Extract per-axis RMSE if present
    per_axis_rmse = None
    if all_results and "_per_axis_rmse" in all_results[-1]:
        per_axis_rmse = all_results[-1]["_per_axis_rmse"]
        all_results = all_results[:-1]  # Remove the special entry
    
    results_df = pd.DataFrame(all_results)

    # Difference columns
    results_df["pred_diff_filtered"]  = results_df["pred_err_all"]  - results_df["pred_err_filtered"]
    results_df["rot_diff_filtered"]   = results_df["rot_err_all"]   - results_df["rot_err_filtered"]
    results_df["trans_diff_filtered"] = results_df["trans_err_all"] - results_df["trans_err_filtered"]

    # results_df["pred_diff_weighted"]  = results_df["pred_err_all"]  - results_df["pred_err_weighted"]
    results_df["rot_diff_weighted_ver1"]   = results_df["rot_err_all"]   - results_df["rot_err_weighted_ver1"]
    results_df["trans_diff_weighted_ver1"] = results_df["trans_err_all"] - results_df["trans_err_weighted_ver1"]

    results_df["rot_diff_weighted_ver2"]   = results_df["rot_err_all"]   - results_df["rot_err_weighted_ver2"]
    results_df["trans_diff_weighted_ver2"] = results_df["trans_err_all"] - results_df["trans_err_weighted_ver2"]

    results_df["rot_diff_weighted_ver3"]   = results_df["rot_err_all"]   - results_df["rot_err_weighted_ver3"]
    results_df["trans_diff_weighted_ver3"] = results_df["trans_err_all"] - results_df["trans_err_weighted_ver3"]

    results_df["pred_diff_gt"] = results_df["pred_err_all"] - results_df["pred_err_gt"]
    results_df["rot_diff_gt"]  = results_df["rot_err_all"]  - results_df["rot_err_gt"]
    results_df["trans_diff_gt"] = results_df["trans_err_all"] - results_df["trans_err_gt"]

    # Reorder columns
    # results_df = results_df[
    #     [
    #         "image",
    #         "pred_err_all",
    #         "pred_err_filtered",
    #         "pred_diff",
    #         "rot_err_all",
    #         "rot_err_filtered",
    #         "rot_diff",
    #         "trans_err_all",
    #         "trans_err_filtered",
    #         "trans_diff",
    #         "pred_better",
    #         "pose_better",
    #     ]
    # ]
            # all_results.append({
            #     "image": image_name,
            #     "pred_err_all": pred_err_all,
            #     "pred_err_filtered": pred_err_filtered,
            #     "pred_better": pred_better,

            #     "rot_err_all": rot_err_all,
            #     "rot_err_filtered": rot_err_filt,
            #     "rot_err_weighted": rot_err_w,

            #     "trans_err_all": trans_err_all,
            #     "trans_err_filtered": trans_err_filt,
            #     "trans_err_weighted": trans_err_w,
                
            #     "pose_better": pose_better,
            #     "pose_better_weighted": pose_better_w,
            # })

    # Build column list with per-axis columns
    base_columns = [
        "image",
        "pred_err_all",
        "pred_err_filtered",
        "pred_diff_filtered",
        "pred_err_gt",
        "pred_diff_gt",

        "rot_err_all",
        "rot_err_all_axis_0",
        "rot_err_all_axis_1",
        "rot_err_all_axis_2",
        "rot_err_filtered",
        "rot_err_filtered_axis_0",
        "rot_err_filtered_axis_1",
        "rot_err_filtered_axis_2",
        "rot_diff_filtered",
        "rot_err_weighted_ver1",
        "rot_err_weighted_ver1_axis_0",
        "rot_err_weighted_ver1_axis_1",
        "rot_err_weighted_ver1_axis_2",
        "rot_diff_weighted_ver1",
        "rot_err_weighted_ver2",
        "rot_err_weighted_ver2_axis_0",
        "rot_err_weighted_ver2_axis_1",
        "rot_err_weighted_ver2_axis_2",
        "rot_diff_weighted_ver2",
        "rot_err_weighted_ver3",
        "rot_err_weighted_ver3_axis_0",
        "rot_err_weighted_ver3_axis_1",
        "rot_err_weighted_ver3_axis_2",
        "rot_diff_weighted_ver3",
        "rot_err_gt",
        "rot_err_gt_axis_0",
        "rot_err_gt_axis_1",
        "rot_err_gt_axis_2",
        "rot_diff_gt",

        "trans_err_all",
        "trans_err_filtered",
        "trans_diff_filtered",
        "trans_err_weighted_ver1",
        "trans_diff_weighted_ver1",
        "trans_err_weighted_ver2",
        "trans_diff_weighted_ver2",
        "trans_err_weighted_ver3",
        "trans_diff_weighted_ver3",
        "trans_err_gt",
        "trans_diff_gt",
    ]
    
    # Keep the historical ordering first, but preserve new metadata columns as well.
    available_columns = [col for col in base_columns if col in results_df.columns]
    extra_columns = [col for col in results_df.columns if col not in available_columns]
    results_df = results_df[available_columns + extra_columns]
    
    # Add per-axis RMSE summary rows if available
    if per_axis_rmse is not None:
        rmse_rows = []
        for key, rmse_dict in per_axis_rmse.items():
            if not pd.isna(rmse_dict['all_axes']):
                row = {"image": f"RMSE_{key}"}
                # Add RMSE values for each rotation error type
                if key == "all":
                    row["rot_err_all"] = rmse_dict['all_axes']
                    row["rot_err_all_axis_0"] = rmse_dict['axis_0']
                    row["rot_err_all_axis_1"] = rmse_dict['axis_1']
                    row["rot_err_all_axis_2"] = rmse_dict['axis_2']
                elif key == "filtered":
                    row["rot_err_filtered"] = rmse_dict['all_axes']
                    row["rot_err_filtered_axis_0"] = rmse_dict['axis_0']
                    row["rot_err_filtered_axis_1"] = rmse_dict['axis_1']
                    row["rot_err_filtered_axis_2"] = rmse_dict['axis_2']
                elif key == "weighted_ver1":
                    row["rot_err_weighted_ver1"] = rmse_dict['all_axes']
                    row["rot_err_weighted_ver1_axis_0"] = rmse_dict['axis_0']
                    row["rot_err_weighted_ver1_axis_1"] = rmse_dict['axis_1']
                    row["rot_err_weighted_ver1_axis_2"] = rmse_dict['axis_2']
                elif key == "weighted_ver2":
                    row["rot_err_weighted_ver2"] = rmse_dict['all_axes']
                    row["rot_err_weighted_ver2_axis_0"] = rmse_dict['axis_0']
                    row["rot_err_weighted_ver2_axis_1"] = rmse_dict['axis_1']
                    row["rot_err_weighted_ver2_axis_2"] = rmse_dict['axis_2']
                elif key == "weighted_ver3":
                    row["rot_err_weighted_ver3"] = rmse_dict['all_axes']
                    row["rot_err_weighted_ver3_axis_0"] = rmse_dict['axis_0']
                    row["rot_err_weighted_ver3_axis_1"] = rmse_dict['axis_1']
                    row["rot_err_weighted_ver3_axis_2"] = rmse_dict['axis_2']
                elif key == "gt":
                    row["rot_err_gt"] = rmse_dict['all_axes']
                    row["rot_err_gt_axis_0"] = rmse_dict['axis_0']
                    row["rot_err_gt_axis_1"] = rmse_dict['axis_1']
                    row["rot_err_gt_axis_2"] = rmse_dict['axis_2']
                rmse_rows.append(row)
        
        if rmse_rows:
            ordered_columns = list(results_df.columns)
            aligned_rmse_rows = [
                {col: row.get(col, None) for col in ordered_columns}
                for row in rmse_rows
            ]
            combined_rows = results_df.to_dict("records") + aligned_rmse_rows
            results_df = pd.DataFrame(combined_rows, columns=ordered_columns)

    
    csv_path = os.path.join(
        args.vis_dir,
        args.save_folder_name,
        "final_results",
        f"test_results_summary_{args.top_k_landmarks}_{suffix}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print("[Saved results CSV] →", csv_path)
    print()

    return results_df
