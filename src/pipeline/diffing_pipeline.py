"""
Diffing pipeline for orchestrating model comparison methods.
"""

from typing import Dict, Any, List
from omegaconf import DictConfig
from loguru import logger

from .pipeline import Pipeline
from src.diffing.methods.kl import KLDivergenceDiffingMethod


class DiffingPipeline(Pipeline):
    """
    Pipeline for running diffing methods to analyze differences between models.
    
    This pipeline can orchestrate multiple diffing methods and chain them together.
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="DiffingPipeline")
        
        # Store diffing configuration
        self.diffing_cfg = cfg.diffing
        
        # Initialize diffing method
        self.diffing_method = self._create_diffing_method()
        
    def _create_diffing_method(self):
        """Create the appropriate diffing method based on configuration."""
        method_name = self.diffing_cfg.method.name
        
        if method_name == "kl_divergence":
            return KLDivergenceDiffingMethod(self.cfg)
        else:
            raise ValueError(f"Unknown diffing method: {method_name}")
    
    def validate_config(self) -> None:
        """Validate the diffing pipeline configuration."""
        super().validate_config()
        
        # Check diffing-specific configuration
        if not hasattr(self.cfg, "diffing"):
            raise ValueError("Configuration must contain 'diffing' section")
            
        if not hasattr(self.cfg.diffing, "method"):
            raise ValueError("Diffing configuration must contain 'method' section")
            
        if not hasattr(self.cfg.diffing.method, "name"):
            raise ValueError("Diffing method configuration must contain 'name'")
            
        # Check model configuration
        if not hasattr(self.cfg, "model"):
            raise ValueError("Configuration must contain 'model' section")
            
        if not hasattr(self.cfg, "organism"):
            raise ValueError("Configuration must contain 'organism' section")
            
        self.logger.info("Diffing pipeline configuration validation passed")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the diffing pipeline.
        
        Returns:
            Dictionary containing results from all diffing methods
        """
        self.logger.info(f"Running diffing method: {self.diffing_cfg.method.name}")
        
        # Run the diffing method
        method_results = self.diffing_method.execute()
        
        # Package results
        results = {
            "method_name": self.diffing_cfg.method.name,
            "method_results": method_results,
            "summary": {
                "total_tokens_processed": method_results.get("metadata", {}).get("total_tokens_processed", 0),
                "base_model": method_results.get("metadata", {}).get("base_model", "unknown"),
                "finetuned_model": method_results.get("metadata", {}).get("finetuned_model", "unknown"),
                "dataset": method_results.get("metadata", {}).get("dataset", "unknown")
            }
        }
        
        # Log summary statistics
        if "statistics" in method_results:
            stats = method_results["statistics"]
            self.logger.info("KL Divergence Statistics:")
            self.logger.info(f"  Mean: {stats.get('mean', 'N/A'):.6f}")
            self.logger.info(f"  Std: {stats.get('std', 'N/A'):.6f}")
            self.logger.info(f"  Min: {stats.get('min', 'N/A'):.6f}")
            self.logger.info(f"  Max: {stats.get('max', 'N/A'):.6f}")
            self.logger.info(f"  Median: {stats.get('median', 'N/A'):.6f}")
        
        # Log max activating examples info
        if "max_activating_examples" in method_results:
            num_examples = len(method_results["max_activating_examples"])
            self.logger.info(f"Captured {num_examples} max activating examples")
            
            if num_examples > 0:
                top_example = method_results["max_activating_examples"][0]
                self.logger.info(f"Highest KL divergence: {top_example['max_kl_value']:.6f}")
        
        return results 