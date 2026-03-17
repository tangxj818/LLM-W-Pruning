import argparse
import os
import torch
import json
import shutil
import glob
from safetensors.torch import safe_open, save_file, load_file
from typing import Dict, List


def load_safetensors(file_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not exists: {file_path}")
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            weights = {k: f.get_tensor(k) for k in f.keys()}
        return weights
    except Exception as e:
        raise RuntimeError(f"Load safetensors failed: {e}")


def load_safetensors_from_folder(folder_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path is not a directory: {folder_path}")

    safetensors_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.safetensors') and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {folder_path}")

    index_files = [
        f for f in os.listdir(folder_path)
        if 'safetensors.index.json' in f and os.path.isfile(os.path.join(folder_path, f))
    ]

    if len(index_files) == 0:
        if len(safetensors_files) != 1:
            raise ValueError(
                f"Expected exactly one .safetensors file when no index is present, "
                f"but found {len(safetensors_files)}: {safetensors_files}"
            )
        file_path = os.path.join(folder_path, safetensors_files[0])
        return load_safetensors(file_path)

    elif len(index_files) == 1:
        index_path = os.path.join(folder_path, index_files[0])
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        weight_map = index_data.get('weight_map', {})
        shard_files = list(dict.fromkeys(weight_map.values()))  

        shard_paths = [os.path.join(folder_path, fname) for fname in shard_files]

        for sp in shard_paths:
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Missing shard file: {sp}")

        weights = {}
        for shard_path in shard_paths:
            shard_weights = load_file(shard_path) 
            weights.update(shard_weights)
        return weights

    else:
        raise ValueError(f"Multiple index files found: {index_files}. Expected at most one.")


def detect_layer_prefix_and_config(config_dict: dict, weights: Dict[str, torch.Tensor]) -> str:
    num_layers = None
    text_config = config_dict.get("text_config", {})
    for key in ["num_hidden_layers", "n_layer", "num_layers"]:
        if key in text_config:
            num_layers = text_config[key]
            break

    if num_layers is None:
        for key in ["num_hidden_layers", "n_layer", "num_layers"]:
            if key in config_dict:
                num_layers = config_dict[key]
                break

    if num_layers is None:
        raise ValueError(
            "Cannot find layer count in config.json. "
            "Check if 'num_hidden_layers'/'n_layer'/'num_layers' exists in root or 'text_config' subdict."
        )

    candidates = [
        "model.layers.",
        "transformer.h.",
        "encoder.layer.",
        "decoder.block.",
        "gpt_neox.layers.",
        "blocks.",
        "model.language_model.layers.",
    ]

    # Check hardcoded candidates first
    for prefix in candidates:
        layer_count = sum(1 for i in range(num_layers) if any(k.startswith(f"{prefix}{i}.") for k in weights))
        if layer_count > 0:
            # Only return this prefix if it has a significant portion of the expected layers
            if layer_count >= num_layers * 0.5:  # At least 50% of layers
                return prefix

    # Fallback: search for any prefix with layer pattern
    prefix_counts = {}
    for k in weights:
        for i in range(num_layers):
            pattern = f".{i}."
            if pattern in k:
                parts = k.split(".")
                for j, part in enumerate(parts):
                    if part == str(i) and j > 0:
                        guessed = ".".join(parts[:j]) + "."
                        prefix_counts[guessed] = prefix_counts.get(guessed, 0) + 1
                        break
    
    if prefix_counts:
        # Return the prefix that appears most frequently
        best_prefix = max(prefix_counts, key=prefix_counts.get)
        return best_prefix
    
    raise ValueError("Could not auto-detect layer prefix from weights.")


def filter_layers_weights(
    weights: Dict[str, torch.Tensor],
    layer_indices: List[int],
    layer_prefix: str,
    keep_base: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Keep only the specified layers (by original index) and optionally base weights.
    Layer keys are NOT renamed.
    """
    layer_indices_set = set(layer_indices)
    max_layer = max(layer_indices)

    missing = []
    for idx in layer_indices:
        old_prefix = f"{layer_prefix}{idx}"
        if not any(k.startswith(old_prefix) for k in weights):
            missing.append(idx)
    if missing:
        raise ValueError(f"No weights found for layers: {missing}")

    filtered = {}

    for k, v in weights.items():
        if k.startswith(layer_prefix):
            try:
                after_prefix = k[len(layer_prefix):]
                layer_idx_str = after_prefix.split('.')[0]
                layer_idx = int(layer_idx_str)
                if layer_idx in layer_indices_set:
                    filtered[k] = v
            except (ValueError, IndexError):
                continue
        elif keep_base:
            filtered[k] = v

    return filtered


def save_layer_safetensors(
    weights: Dict[str, torch.Tensor],
    output_path: str,
    metadata: dict = None
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    final_metadata = {"framework": "pt"}
    if metadata:
        final_metadata.update(metadata)

    try:
        save_file(weights, output_path, metadata=final_metadata)
        print(f"Saved {len(weights)} weights to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Save safetensors failed: {e}")


def parse_layers_arg(layers_str: str) -> List[int]:
    """Parse layer argument like '0-3' or '0,1,2,3' or '0,2,4'"""
    layers_str = layers_str.strip()
    if '-' in layers_str and ',' not in layers_str:
        start, end = map(int, layers_str.split('-'))
        if start > end:
            raise ValueError(f"Invalid layer range: {layers_str}")
        return list(range(start, end + 1))
    elif ',' in layers_str:
        return [int(x.strip()) for x in layers_str.split(',') if x.strip()]
    else:
        return [int(layers_str)]


def main():
    parser = argparse.ArgumentParser(description="Extract specific layers (e.g., 0-3) from HF model")
    parser.add_argument("--input-dir", "-d", required=True, help="Input HuggingFace model directory")
    parser.add_argument("--output-dir", "-o", required=True, help="Output HuggingFace model directory")
    parser.add_argument("--layers", "-l", required=True, type=str, 
                        help="Layer indices to extract, e.g. '0-3' or '0,1,2,3'")
    parser.add_argument("--keep-base", action="store_true", default=True,
                        help="Keep base weights (embeddings, lm_head, etc.). Default: True.")
    args = parser.parse_args()
    
    try:
        input_dir = os.path.abspath(args.input_dir)
        output_dir = os.path.abspath(args.output_dir)

        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        selected_layers = parse_layers_arg(args.layers)
        print(f"Selected layers: {selected_layers}")

        print(f"Copying entire directory from {input_dir} to {output_dir}...")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(input_dir, output_dir)
        print("Copy completed.")

        config_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {input_dir}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"Loading weights from original directory: {input_dir}")
        raw_weights = load_safetensors_from_folder(input_dir)
        print(f"Total weights loaded: {len(raw_weights)}")

        layer_prefix = detect_layer_prefix_and_config(config, raw_weights)
        print(f"Detected layer prefix: '{layer_prefix}'")

        num_layers_orig = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers") or config.get("text_config", {}).get("num_hidden_layers")
        if num_layers_orig is None:
            raise ValueError("Cannot find layer count in config.json")

        for idx in selected_layers:
            if not (0 <= idx < num_layers_orig):
                raise ValueError(f"Layer index {idx} out of range [0, {num_layers_orig})")

        print(f"Filtering layers: {selected_layers}...")
        filtered_weights = filter_layers_weights(
            raw_weights,
            layer_indices=selected_layers,
            layer_prefix=layer_prefix,
            keep_base=args.keep_base
        )
        print(f"Filtered weights count: {len(filtered_weights)}")

        print("Removing old weight files...")
        for f in glob.glob(os.path.join(output_dir, "*.safetensors")):
            os.remove(f)
        for f in glob.glob(os.path.join(output_dir, "*safetensors.index.json")):
            os.remove(f)

        new_weight_path = os.path.join(output_dir, "model.safetensors")
        save_layer_safetensors(filtered_weights, new_weight_path)

        new_num_layers = len(selected_layers)
        if new_num_layers == 0:
            raise ValueError("selected_layers 不能为空，请选择至少1层")
        
        if "text_config" in config:
            config["text_config"]["num_hidden_layers"] = new_num_layers
            if "n_layer" in config["text_config"]:
                config["text_config"]["n_layer"] = new_num_layers
            if "num_layers" in config["text_config"]:
                config["text_config"]["num_layers"] = new_num_layers

            if "layer_types" in config["text_config"]:
                original_layer_types = config["text_config"]["layer_types"]
                original_num_layers = config["text_config"].get("num_hidden_layers", len(original_layer_types))
                config["text_config"]["layer_types"] = original_layer_types[:new_num_layers]
        else:
            config["num_hidden_layers"] = new_num_layers
            if "n_layer" in config:
                config["n_layer"] = new_num_layers
            if "num_layers" in config:
                config["num_layers"] = new_num_layers

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Updated config.json with num_hidden_layers={new_num_layers}")

        print(f"Model with layers {selected_layers} ready at: {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()