import json
import logging
import os
import re
import yaml
from typing import List
from pathlib import Path


def create_directories(args) -> List[str]:
    """
    Create directories for storing model results based on the selected models.
    
    Args:
        args: Arguments containing model selection and base directory paths
    
    Returns:
        List[str]: List of full directory paths that were created or processed
    """
    # Define model groups
    model_groups = {
        'vlm': {
            'base_dir': os.path.join(args.results_dir, "vlm"),
            'models': ['cloud_ru', 'ollama']
        }
    }

    processed_paths = []
    
    def create_model_dir(base_dir: str, model: str) -> None:
        """Helper function to create directory for a single model"""
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Created base directory: {base_dir}")
            
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created model directory: {model_dir}")
        else:
            print(f"Directory already exists: {model_dir}")
        processed_paths.append(model_dir)
    
    def process_model(model_name: str) -> bool:
        """Process a single model or group name, return True if found"""
        # Check if it's "all"
        if model_name.lower() == "all":
            for group in model_groups.values():
                for model in group['models']:
                    create_model_dir(group['base_dir'], model)
            return True

        # Check if it's a group name
        if model_name.lower() in model_groups:
            group_info = model_groups[model_name.lower()]
            for model in group_info['models']:
                create_model_dir(group_info['base_dir'], model)
            return True
            
        # Check if it's a specific model name
        for group_info in model_groups.values():
            if model_name.lower() in [m.lower() for m in group_info['models']]:
                create_model_dir(group_info['base_dir'], model_name)
                return True
                
        return False

    # Handle single model or list of models
    selected_models = args.model if isinstance(args.model, list) else [args.model]
    
    for model in selected_models:
        if not process_model(model):
            print(f"Warning: No matching model or group found for '{model}'")
            continue
    
    if not processed_paths:
        print("No valid models were processed")
        import sys
        sys.exit(1)
    
    return processed_paths

def setup_logging(path: str, current_run: str) -> logging.Logger:
    """
    Set up logging for each model directory with separate loggers.
    """
    model_name = os.path.basename(path)
    current_run_dir = os.path.join(path, current_run)
    os.makedirs(current_run_dir, exist_ok=True)
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        for handler in logger.handlers[:]:  # Copy to avoid modifying during iteration
            handler.close()  # Close handlers before clearing
            logger.removeHandler(handler)
            
    logfile = os.path.join(current_run_dir, "logfile.log")
    fh = logging.FileHandler(logfile, "w")
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter("%(asctime)s %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, current_run_dir

def load_yaml_config(file_path: str) -> dict:
    yaml_path = Path(file_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def calculate_num_occurrences(ground_truth: str, ocr_text: str):
    """
    Calculate the number of occurrences of words from ground_truth
    """
    key_words = re.findall(r'\w+', ground_truth.lower())
    ocr_tokens = re.findall(r'\w+', ocr_text.lower())
    ocr_set = set(ocr_tokens)  # O(len(ocr_tokens))
    num_occurrences = 0
    for word in key_words:
        if word in ocr_set:
            num_occurrences += 1
    total_words = len(key_words)

    return num_occurrences, total_words


def make_summary(current_run, results_dir) -> List:
    summary = []
    
    for model_result in os.listdir(results_dir):
        for model in os.listdir(os.path.join(results_dir,model_result)):
            num_occurrences = 0
            num_key_words = 0
            total_frames = 0
            total_vids = 0

            for run in os.listdir(os.path.join(results_dir,model_result, model)):
                
                if current_run in run:
                    for evals in os.listdir(os.path.join(results_dir,model_result, model,run,"evaluations")):
                        total_vids+=1
                        
                        json_path = os.path.join(results_dir,model_result, model,run,"evaluations",evals)
                        
                        with open(json_path,"r") as f:
                            json_data = json.load(f)
                            
                        for entry in json_data:
                            num_occurrences += entry["num_occurrences"]
                            num_key_words += entry["num_key_words"]
                            total_frames += 1

            if total_vids!=0:
                summary.append(
                    {
                        "model" : model,
                        "total_vids" : total_vids,
                        "total_frames" : total_frames,
                        "num_occurrences": num_occurrences,
                        "num_key_words": num_key_words,
                        "avg_acc" : num_occurrences / num_key_words,
                    }
                )
    return summary
