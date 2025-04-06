# generate_concept_drift.py
import numpy as np
from collections import Counter
import os
from pathlib import Path
import yaml
import argparse
import torchvision
import torchvision.transforms as transforms
from PIL import Image

DATASET_LOADER = 'torchvision'

# --- (Include the simulate_concept_drift function here - use the improved version) ---
def simulate_concept_drift(original_labels, target_distribution, total_samples=None):
    """
    Simulates concept drift by changing class distribution.
    Handles calculation of remaining proportions if not all classes are specified.
    """
    num_classes_data = len(np.unique(original_labels)) # Classes actually in data
    defined_proportion = sum(target_distribution.values())
    defined_classes = len(target_distribution)
    # Calculate undefined based on actual classes present in data
    undefined_classes = len([cls for cls in np.unique(original_labels) if cls not in target_distribution])


    if not np.isclose(defined_proportion, 1.0):
         if defined_proportion < 1.0 and undefined_classes > 0:
              print(f"Distributing remaining proportion {(1.0 - defined_proportion):.4f} among {undefined_classes} unspecified classes.")
              remaining_proportion_per_class = (1.0 - defined_proportion) / undefined_classes
              # Add proportion only for classes present in the data but not specified in target_dist
              present_labels = np.unique(original_labels)
              for i in present_labels:
                   if i not in target_distribution:
                        target_distribution[i] = remaining_proportion_per_class
         if not np.isclose(sum(target_distribution.values()), 1.0):
              # Recalculate sum after potential additions
               raise ValueError(f"Target distribution proportions must sum to 1.0. Current sum: {sum(target_distribution.values())}")

    elif defined_classes < num_classes_data and defined_proportion == 1.0 :
         # Sum is 1.0, but not all classes defined - set others to 0 proportion
         print("Warning: Target distribution sums to 1.0, but not all classes defined. Setting undefined classes proportion to 0.")
         present_labels = np.unique(original_labels)
         for i in present_labels:
              if i not in target_distribution:
                   target_distribution[i] = 0.0

    if total_samples is None:
        total_samples = len(original_labels)

    original_labels = np.array(original_labels)

    unique_original_labels = np.unique(original_labels)
    original_indices_by_class = {
        cls: np.where(original_labels == cls)[0]
        for cls in unique_original_labels
    }

    drifted_indices = []
    actual_counts = Counter()

    for class_index, proportion in target_distribution.items():
        if proportion <= 0: continue # Skip classes with 0 or negative proportion

        # Check if class exists in data, even if proportion is > 0
        if class_index not in original_indices_by_class or len(original_indices_by_class[class_index]) == 0:
            print(f"Warning: Class {class_index} requested (proportion {proportion}), "
                  f"but no samples found in original data. Skipping this class.")
            continue

        num_samples_for_class = int(np.round(total_samples * proportion))
        if num_samples_for_class == 0:
             # This can happen if total_samples * proportion rounds down to 0
             print(f"Warning: Proportion {proportion} for class {class_index} resulted in 0 samples requested. Skipping.")
             continue

        available_indices = original_indices_by_class[class_index]

        replace = num_samples_for_class > len(available_indices)
        if replace:
             print(f"Warning: Requested {num_samples_for_class} samples for class {class_index}, "
                   f"but only {len(available_indices)} available. Sampling with replacement.")

        chosen_indices = np.random.choice(
            available_indices, size=num_samples_for_class, replace=replace
        )
        drifted_indices.extend(chosen_indices)
        actual_counts[class_index] = len(chosen_indices)

    drifted_indices = np.array(drifted_indices)

    if len(drifted_indices) == 0:
        print("Error: No indices were selected for the drifted dataset. Check target distribution and data.")
        return np.array([]), {}

    np.random.shuffle(drifted_indices)

    total_drifted = len(drifted_indices)
    # Report distribution based on classes present in the original data
    actual_distribution = {cls: actual_counts.get(cls, 0) / total_drifted for cls in unique_original_labels}


    print(f"Full target distribution requested/calculated: {target_distribution}")
    print(f"Actual distribution achieved ({total_drifted} samples): {actual_distribution}")

    return drifted_indices, actual_distribution


def main(config_path):
    # --- Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config.get('dataset', {})
    output_config = config.get('output', {})
    # <<< CHANGE HERE: Load only concept drift scenarios >>>
    scenarios = config.get('concept_drift_scenarios', {})

    dataset_root = dataset_config.get('root_path', './data')
    base_image_dir = Path(output_config.get('base_image_dir', 'simulated_drift_images'))
    base_image_dir.mkdir(parents=True, exist_ok=True)

    # --- Load CIFAR-10 Data ---
    global DATASET_LOADER
    cifar_images = None
    cifar_labels = None
    if DATASET_LOADER == 'torchvision':
        try:
            # Load train dataset for concept drift simulation
            full_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True)
            cifar_images = full_dataset.data
            cifar_labels = np.array(full_dataset.targets)
            print(f"Loaded CIFAR-10 train data: {cifar_images.shape}, {len(cifar_labels)} labels from {dataset_root}.")
        except Exception as e:
            print(f"Error loading CIFAR-10 with torchvision from {dataset_root}: {e}")
            DATASET_LOADER = None
    # Add TensorFlow loading here if needed

    if DATASET_LOADER is None or cifar_images is None or cifar_labels is None:
        print("Could not load CIFAR-10 data. Exiting.")
        return

    # --- Process Each Scenario Defined in Config ---
    if not scenarios:
         print("No concept drift scenarios defined in the configuration file.")
         return

    for scenario_name, scenario_config in scenarios.items():
        print(f"\n--- Processing Concept Drift Scenario: {scenario_name} ---")

        target_distribution = scenario_config.get('target_distribution', {})
        total_samples = scenario_config.get('total_samples', None)

        if not target_distribution:
            print(f"Skipping scenario '{scenario_name}': No target_distribution defined.")
            continue

        print(f"Simulating concept drift for {scenario_name}...")
        drifted_indices, achieved_dist = simulate_concept_drift(
            cifar_labels,
            target_distribution.copy(), # Pass a copy
            total_samples=total_samples
        )

        if len(drifted_indices) == 0:
            print(f"Skipping image saving for scenario '{scenario_name}' as no indices were generated.")
            continue

        # --- Save selected images to files ---
        scenario_dir = base_image_dir / scenario_name # Use scenario name for folder
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving {len(drifted_indices)} images for {scenario_name} to {scenario_dir}...")
        saved_count = 0
        for i, idx in enumerate(drifted_indices):
            try:
                image_array = cifar_images[idx]
                img = Image.fromarray(image_array)
                original_label = cifar_labels[idx]
                filename = f"img_{i:04d}_origlabel_{original_label}_idx_{idx}.png"
                save_path = scenario_dir / filename
                img.save(save_path)
                saved_count += 1
            except Exception as e:
                print(f"Error saving image at original index {idx} for scenario {scenario_name}: {e}")

        print(f"Finished saving {saved_count} images for concept drift scenario {scenario_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate concept drifted image datasets based on a config file.")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)"
    )
    args = parser.parse_args()
    main(args.config)