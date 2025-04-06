# generate_data_drift.py
import numpy as np
import os
from pathlib import Path
import yaml
import argparse
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageOps
import random

DATASET_LOADER = 'torchvision'

# --- (Include helper functions apply_brightness, apply_rotation, apply_gaussian_noise here) ---
def apply_brightness(img_pil, factor):
    enhancer = ImageEnhance.Brightness(img_pil)
    return enhancer.enhance(factor)

def apply_rotation(img_pil, degrees):
    angle = random.uniform(-degrees, degrees)
    # Fill with average color might be better than black/white for rotation
    # avg_color = tuple(np.array(img_pil).mean(axis=(0,1)).astype(int)) # Requires numpy
    return img_pil.rotate(angle, expand=False) # fillcolor=avg_color

def apply_gaussian_noise(img_pil, mean=0.0, std=0.1):
     img_np = np.array(img_pil) / 255.0
     noise = np.random.normal(mean, std, img_np.shape)
     noisy_img_np = np.clip(img_np + noise, 0, 1)
     noisy_img_pil = Image.fromarray((noisy_img_np * 255).astype(np.uint8))
     return noisy_img_pil

# --- Main Simulation Function ---
def main(config_path):
    # --- Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config.get('dataset', {})
    output_config = config.get('output', {})
    # <<< CHANGE HERE: Load only data drift scenarios >>>
    scenarios = config.get('data_drift_scenarios', {})

    dataset_root = dataset_config.get('root_path', './data')
    base_image_dir = Path(output_config.get('base_image_dir', 'simulated_drift_images'))
    base_image_dir.mkdir(parents=True, exist_ok=True)

    # --- Load CIFAR-10 Data ---
    global DATASET_LOADER
    datasets = {}
    if DATASET_LOADER == 'torchvision':
        try:
            train_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True)
            test_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True)
            datasets = {'train': train_dataset, 'test': test_dataset}
            print(f"Loaded CIFAR-10 train/test data from {dataset_root}.")
        except Exception as e:
            print(f"Error loading CIFAR-10 with torchvision from {dataset_root}: {e}")
            DATASET_LOADER = None
    # Add TensorFlow loading here if needed

    if DATASET_LOADER is None or not datasets:
        print("Could not load CIFAR-10 data. Exiting.")
        return

    # --- Process Each Scenario ---
    if not scenarios:
         print("No data drift scenarios defined in the configuration file.")
         return

    for scenario_name, scenario_config in scenarios.items():
        print(f"\n--- Processing Data Drift Scenario: {scenario_name} ---")

        drift_type = scenario_config.get('type')
        num_samples = scenario_config.get('num_samples', 100)
        base_ds_name = scenario_config.get('base_dataset', 'train')

        if base_ds_name not in datasets:
            print(f"Error: Base dataset '{base_ds_name}' not loaded. Skipping scenario.")
            continue
        if not drift_type:
             print(f"Error: No 'type' specified for data drift scenario '{scenario_name}'. Skipping.")
             continue

        base_dataset = datasets[base_ds_name]
        dataset_size = len(base_dataset)

        if num_samples > dataset_size:
            print(f"Warning: Requested {num_samples} samples, but base dataset '{base_ds_name}' only has {dataset_size}. Using all samples.")
            indices_to_sample = np.arange(dataset_size)
        else:
            indices_to_sample = np.random.choice(dataset_size, num_samples, replace=False)

        scenario_dir = base_image_dir / scenario_name # Use scenario name for folder
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"Applying '{drift_type}' drift and saving {len(indices_to_sample)} images to {scenario_dir}...")
        saved_count = 0
        for i, idx in enumerate(indices_to_sample):
            try:
                image_data, original_label = base_dataset[idx]
                if not isinstance(image_data, Image.Image):
                    if isinstance(image_data, torch.Tensor):
                         img_pil = transforms.ToPILImage()(image_data)
                    else: # Assume numpy HWC
                         img_pil = Image.fromarray(image_data)
                else:
                     img_pil = image_data

                # Apply transformation
                drifted_img_pil = None
                if drift_type == 'brightness':
                    factor = scenario_config.get('factor', 1.0)
                    drifted_img_pil = apply_brightness(img_pil, factor)
                elif drift_type == 'rotation':
                    degrees = scenario_config.get('degrees', 0)
                    drifted_img_pil = apply_rotation(img_pil, degrees)
                elif drift_type == 'gaussian_noise':
                    mean = scenario_config.get('mean', 0.0)
                    std = scenario_config.get('std', 0.1)
                    drifted_img_pil = apply_gaussian_noise(img_pil, mean, std)
                else:
                    print(f"Warning: Unknown drift type '{drift_type}'. Saving original image.")
                    drifted_img_pil = img_pil

                # Save the drifted image
                filename = f"img_{i:04d}_origlabel_{original_label}_idx_{idx}.png"
                save_path = scenario_dir / filename
                drifted_img_pil.save(save_path)
                saved_count += 1

            except Exception as e:
                print(f"Error processing image at original index {idx} for scenario {scenario_name}: {e}")
                # Consider adding traceback: import traceback; traceback.print_exc()

        print(f"Finished saving {saved_count} drifted images for data drift scenario {scenario_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data drifted image datasets based on a config file.")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    args = parser.parse_args()
    main(args.config)