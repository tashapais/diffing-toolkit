"""
Streamlit dashboard for visualizing model diffing results.

This dashboard dynamically discovers available model organisms and diffing methods
from the filesystem and provides an interactive interface to explore the results.
"""

import streamlit as st
from omegaconf import DictConfig, OmegaConf
import sys
from typing import Dict, List, Tuple
import os
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
import time
import torch

from src.pipeline.diffing_pipeline import get_method_class  

# # Disable Streamlit file watcher to avoid torch._classes compatibility issues
# os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


def get_main_cfg() -> DictConfig:
    """Load the main configuration using Hydra."""
    
    # Get absolute path to configs directory
    config_dir = Path("configs").resolve()
    
    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra with the configs directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Load the main config (assuming config.yaml exists)
        cfg = compose(config_name="config")
        return cfg

def discover_organisms() -> List[str]:
    """Discover available organisms from the configs directory."""
    organism_configs = Path("configs/organism").glob("*.yaml")
    return [f.stem for f in organism_configs]


def discover_methods() -> List[str]:
    """Discover available diffing methods from the configs directory."""
    method_configs = Path("configs/diffing/method").glob("*.yaml")
    methods = [f.stem for f in method_configs if f.stem != "example"]
    return methods


@st.cache_data
def get_available_results() -> Dict[str, Dict[str, List[str]]]:
    """
    Compile available results from all diffing methods.
    
    Returns:
        Dict mapping {model: {organism: [methods]}}
    """
    available = {}
    
    main_cfg = get_main_cfg()
    # Get available methods from configs
    available_methods = discover_methods()
    # Check each method for available results
    for method_name in available_methods:
        method_class = get_method_class(method_name)
        print(method_class)
        
        # Call static method directly on the class
        method_results = method_class.has_results(Path(main_cfg.diffing.results_base_dir))
        print(method_results)
        # Compile results into the global structure
        for model_name, organisms in method_results.items():
            if model_name not in available:
                available[model_name] = {}
            
            for organism_name, path in organisms.items():
                if organism_name not in available[model_name]:
                    available[model_name][organism_name] = []
                
                available[model_name][organism_name].append(method_name)

    
    return available

@st.cache_data
def load_config(model: str, organism: str, method: str) -> DictConfig:
    """
    Create minimal config for initializing diffing methods.
    
    Args:
        model: Model name
        organism: Organism name  
        method: Method name
        
    Returns:
        Minimal DictConfig for the method
    """
    # Get absolute path to configs directory
    config_dir = Path("configs").resolve()
    
    # Clear any existing Hydra instance
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"

    # Initialize Hydra with the configs directory
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose config with overwrites for model, organism, and method
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model}",
                f"organism={organism}",
                f"diffing/method={method}",
                f"model.dtype={dtype}"
            ]
        )
        
        # Resolve the configuration to ensure all interpolations are evaluated
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)
        return cfg

def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Model Diffing Dashboard",
        page_icon="🧬",
        layout="wide"
    )
    

    
    st.title("🧬 Model Diffing Dashboard")
    st.markdown("Explore differences between base and finetuned models")
    
    # Discover available results
    available_results = get_available_results()
    if not available_results:
        st.error("No diffing results found. Run some diffing experiments first!")
        return
    
    # Model selection
    available_models = list(available_results.keys())
    selected_model = st.selectbox("Select Base Model", available_models)
    
    if not selected_model:
        return
    
    # Organism selection
    available_organisms = list(available_results[selected_model].keys())
    selected_organism = st.selectbox("Select Organism", available_organisms)
    
    if not selected_organism:
        return
    
    # Method selection
    available_methods = available_results[selected_model][selected_organism]
    if not available_methods:
        st.warning(f"No results found for {selected_model}/{selected_organism}")
        return
    selected_method = st.selectbox(
        "Select Diffing Method", 
        ["Select a method..."] + available_methods,
        index=0
    )
    
    if selected_method == "Select a method...":
        return
    
    # Create and initialize the diffing method
    try:
        start_time = time.time()
        with st.spinner("Loading method..."):
            cfg = load_config(selected_model, selected_organism, selected_method)
            method_class = get_method_class(selected_method)
            
            if method_class is None:
                st.error(f"Unknown method: {selected_method}")
                return
            
            # Initialize method (without loading models for visualization)
            method = method_class(cfg)
            
            # Call the visualize method
            start_time = time.time()
            method.visualize()
        print(f"Method visualization took: {time.time() - start_time:.3f}s")
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main() 