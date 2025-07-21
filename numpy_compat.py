"""
NumPy compatibility patch for dictionary_learning package.
This adds the np.bool attribute that was removed in newer NumPy versions.
"""
import numpy as np

# Add np.bool compatibility for older packages
if not hasattr(np, 'bool'):
    np.bool = np.bool_ 