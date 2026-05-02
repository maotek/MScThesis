"""
Event-based Depth Estimation Evaluation Module

This module provides comprehensive evaluation tools for depth estimation models,
specifically designed for event-based vision datasets. It includes metrics calculation,
data preprocessing, visualization tools, and depth conversion utilities.

The module supports various depth representations (linear, logarithmic, inverse)
and provides standard evaluation metrics used in depth estimation research.

Taken from Depth AnyEvent
"""

import numpy as np
import torch
from pprint import pprint
import matplotlib.pyplot as plt


# Standard depth evaluation metrics for different depth thresholds
# These metrics cover overall evaluation (_) and threshold-specific evaluation (10, 20, 30 meters)
METRICS_KEYWORDS = [
    # Overall metrics
    "_abs_rel_diff",              # Absolute relative difference
    "_squ_rel_diff",              # Squared relative difference  
    "_RMS_linear",                # Root mean square linear error
    "_RMS_log",                   # Root mean square logarithmic error
    "_SILog",                     # Scale-invariant logarithmic error
    "_mean_target_depth",         # Mean ground truth depth
    "_median_target_depth",       # Median ground truth depth
    "_mean_prediction_depth",     # Mean predicted depth
    "_median_prediction_depth",   # Median predicted depth
    "_mean_depth_error",          # Mean absolute depth error
    "_median_diff",               # Median difference between GT and prediction
    "_threshold_delta_1.25",      # Percentage of pixels with ratio < 1.25
    "_threshold_delta_1.25^2",    # Percentage of pixels with ratio < 1.25^2
    "_threshold_delta_1.25^3",    # Percentage of pixels with ratio < 1.25^3
    
    # Metrics for depth threshold 10m
    "_10_mean_target_depth", "_10_median_target_depth",
    "_10_mean_prediction_depth", "_10_median_prediction_depth",
    "_10_abs_rel_diff", "_10_squ_rel_diff", "_10_RMS_linear",
    "_10_RMS_log", "_10_SILog", "_10_mean_depth_error",
    "_10_median_diff", "_10_threshold_delta_1.25",
    "_10_threshold_delta_1.25^2", "_10_threshold_delta_1.25^3",
    
    # Metrics for depth threshold 20m
    "_20_abs_rel_diff", "_20_squ_rel_diff", "_20_RMS_linear",
    "_20_RMS_log", "_20_SILog", "_20_mean_target_depth",
    "_20_median_target_depth", "_20_mean_prediction_depth",
    "_20_median_prediction_depth", "_20_mean_depth_error",
    "_20_median_diff", "_20_threshold_delta_1.25",
    "_20_threshold_delta_1.25^2", "_20_threshold_delta_1.25^3",
    
    # Metrics for depth threshold 30m
    "_30_abs_rel_diff", "_30_squ_rel_diff", "_30_RMS_linear",
    "_30_RMS_log", "_30_SILog", "_30_mean_target_depth",
    "_30_median_target_depth", "_30_mean_prediction_depth",
    "_30_median_prediction_depth", "_30_mean_depth_error",
    "_30_median_diff", "_30_threshold_delta_1.25",
    "_30_threshold_delta_1.25^2", "_30_threshold_delta_1.25^3",
]


def prepare_target_data(target, clip_distance):
    """
    Prepare target (ground truth) depth data for evaluation.
    
    Args:
        target (np.ndarray): Ground truth depth data
        clip_distance (float): Maximum depth value for clipping
        
    Returns:
        np.ndarray: Processed target depth data
    """
    # Clip and normalize target depth
    target = np.clip(target, 0, clip_distance)
    target = target / np.amax(target[~np.isnan(target)])
    
    # Scale to absolute depth values
    target *= clip_distance
    
    return target


def prepare_target_data_torch(target, clip_distance):
    """
    Prepare target depth data using PyTorch tensors (GPU-friendly version).
    
    Args:
        target (torch.Tensor): Ground truth depth data
        clip_distance (float): Maximum depth value for clipping
        
    Returns:
        torch.Tensor: Processed target depth data
    """
    # Clip and normalize target depth
    target = torch.clamp(target, 0, clip_distance)
    target = target / torch.amax(target[~torch.isnan(target)])
    
    # Scale to absolute depth values
    target *= clip_distance
    
    return target


def add_to_metrics(idx, metrics, target_, prediction_, mask, event_frame=None, 
                  prefix="", debug=False, output_folder=None):
    """
    Calculate and accumulate depth evaluation metrics.
    
    Args:
        idx (int): Current frame index
        metrics (dict): Dictionary to accumulate metrics
        target_ (np.ndarray): Ground truth depth data (full image)
        prediction_ (np.ndarray): Predicted depth data (full image)
        mask (np.ndarray): Valid pixel mask
        event_frame (np.ndarray, optional): Event frame for visualization
        prefix (str): Prefix for metric names
        debug (bool): Enable debug visualizations
        output_folder (str, optional): Folder to save debug plots
        
    Returns:
        dict: Updated metrics dictionary
    """
    # Initialize metrics dictionary if empty
    if len(metrics) == 0:
        metrics = {k: 0 for k in METRICS_KEYWORDS}

    # Create prediction and depth validity masks
    max_target = np.amax(target_[~np.isnan(target_)])
    prediction_mask = (prediction_ > 0) & (prediction_ < max_target)
    depth_mask = (target_ > 0) & (target_ < max_target)

    # Combine all masks
    valid_mask = mask & depth_mask & prediction_mask
    if not np.any(valid_mask):
        return metrics
    eps = 1e-5  # Small epsilon to avoid division by zero

    # Extract valid pixels only
    target = target_[valid_mask]
    prediction = prediction_[valid_mask]

    # Calculate threshold metrics (accuracy within certain ratios)
    ratio = np.max(np.stack([target/(prediction+eps), prediction/(target+eps)]), axis=0)

    # Initialize new metrics for this frame
    new_metrics = {}
    
    # Threshold accuracy metrics
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25**2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25**3)

    # Calculate absolute and logarithmic differences
    log_diff = np.log(target + eps) - np.log(prediction + eps)
    abs_diff = np.abs(target - prediction)

    # Error metrics
    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff / (target + eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff**2 / (target**2 + eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff**2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff**2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff**2).mean() - (log_diff.mean())**2
    
    # Statistical metrics
    new_metrics[f"{prefix}mean_target_depth"] = target.mean()
    new_metrics[f"{prefix}median_target_depth"] = np.median(target)
    new_metrics[f"{prefix}mean_prediction_depth"] = prediction.mean()
    new_metrics[f"{prefix}median_prediction_depth"] = np.median(prediction)
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()
    new_metrics[f"{prefix}median_diff"] = np.abs(np.median(target) - np.median(prediction))

    # Accumulate metrics
    for k, v in new_metrics.items():
        metrics[k] += v

    # Debug visualization
    if debug:
        print("Current frame metrics:")
        pprint(new_metrics)
        
        # Create comprehensive debug visualization
        fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(15, 20))
        
        # Row 1: Basic depth visualization
        img_gt = ax[0, 0].imshow(target_, vmin=0, vmax=200, cmap='viridis')
        ax[0, 0].set_title("Ground Truth Depth")
        fig.colorbar(img_gt, ax=ax[0, 0])
        
        img_pred = ax[0, 1].imshow(prediction_, vmin=0, vmax=200, cmap='viridis')
        ax[0, 1].set_title("Predicted Depth")
        fig.colorbar(img_pred, ax=ax[0, 1])
        
        target_debug = target_.copy()
        target_debug[~valid_mask] = 0
        img_gt_mask = ax[0, 2].imshow(target_debug, vmin=0, vmax=200, cmap='viridis')
        ax[0, 2].set_title("Ground Truth (Masked)")
        fig.colorbar(img_gt_mask, ax=ax[0, 2])

        # Row 2: Logarithmic visualization
        img_log_gt = ax[1, 0].imshow(np.log(target_ + eps), vmin=0, vmax=np.log(200), cmap='plasma')
        ax[1, 0].set_title("Log Ground Truth")
        fig.colorbar(img_log_gt, ax=ax[1, 0])
        
        img_log_pred = ax[1, 1].imshow(np.log(prediction_ + eps), vmin=0, vmax=np.log(200), cmap='plasma')
        ax[1, 1].set_title("Log Prediction")
        fig.colorbar(img_log_pred, ax=ax[1, 1])
        
        img_ratio = ax[1, 2].imshow(np.max(np.stack([target_ / (prediction_ + eps), 
                           prediction_ / (target_ + eps)]), axis=0), cmap='jet')
        ax[1, 2].set_title("Max Ratio")
        fig.colorbar(img_ratio, ax=ax[1, 2])

        # Row 3: Error visualization
        img_log_diff = ax[2, 0].imshow(np.abs(np.log(target_ + eps) - np.log(prediction_ + eps)), cmap='hot')
        ax[2, 0].set_title("Absolute Log Difference")
        fig.colorbar(img_log_diff, ax=ax[2, 0])
        
        img_abs_diff = ax[2, 1].imshow(np.abs(target_ - prediction_), cmap='hot')
        ax[2, 1].set_title("Absolute Difference")
        fig.colorbar(img_abs_diff, ax=ax[2, 1])
        
        # Event frame visualization (if available)
        if event_frame is not None:
            event_viz = np.zeros((*event_frame.shape[:2], 3))
            event_viz[:, :, 0] = (np.sum(event_frame.astype("float32"), axis=-1) > 0)
            event_viz[:, :, 1] = np.clip(target_.copy(), 0, 1)
            ax[2, 2].imshow(event_viz)
            ax[2, 2].set_title("Event Frame Overlay")

        # Row 4: Masked error visualization
        log_diff_masked = np.abs(np.log(target_ + eps) - np.log(prediction_ + eps))
        log_diff_masked[~valid_mask] = 0
        img_log_diff_mask = ax[3, 0].imshow(log_diff_masked, cmap='hot')
        ax[3, 0].set_title("Abs Log Diff (Masked)")
        fig.colorbar(img_log_diff_mask, ax=ax[3, 0])
        
        abs_diff_masked = np.abs(target_ - prediction_)
        abs_diff_masked[~valid_mask] = 0
        img_abs_diff_mask = ax[3, 1].imshow(abs_diff_masked, cmap='hot')
        ax[3, 1].set_title("Abs Diff (Masked)")
        fig.colorbar(img_abs_diff_mask, ax=ax[3, 1])
        
        img_valid = ax[3, 2].imshow(valid_mask, cmap='gray')
        ax[3, 2].set_title("Valid Pixel Mask")
        fig.colorbar(img_valid, ax=ax[3, 2])

        plt.tight_layout()
        plt.suptitle(f"{prefix}Depth Evaluation - Frame {idx}")
        
        if output_folder:
            plt.savefig(f"{output_folder}/debug_frame_{idx:06d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return metrics