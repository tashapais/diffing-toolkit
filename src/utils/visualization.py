"""
Shared visualization utilities for diffing methods.

This module provides common functionality for converting diffing results
into HTML visualizations using the tiny-dashboard library.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import torch
import streamlit as st
from pathlib import Path
import sys
import json
from tiny_dashboard.html_utils import (
    create_example_html,
    create_base_html,
    create_highlighted_tokens_html,
)
from transformers import AutoTokenizer
from numpy import array

from src.utils.model import load_tokenizer_from_config
from src.utils.configs import ModelConfig


@st.cache_data
def convert_max_examples_to_dashboard_format(
    max_examples: List[Dict[str, Any]],
    model_cfg: ModelConfig,
) -> List[Tuple[float, List[str], List[float], str]]:
    """
    Convert max_activating_examples from diffing results to dashboard format.

    Args:
        max_examples: List of max activating examples from diffing results
        tokenizer_name: Name/path of the tokenizer for caching

    Returns:
        List of tuples (max_activation_value, tokens, activation_values, text)
    """
    tokenizer = load_tokenizer_from_config(model_cfg)

    dashboard_examples = []

    for example in max_examples:
        max_score = example["max_score"]
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
        scores_per_token = array(example["scores_per_token"])
        scores_per_token = scores_per_token - scores_per_token.min()

        # Get the text for search functionality
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

        dashboard_examples.append((max_score, tokens, scores_per_token, text))

    return dashboard_examples


def create_html_highlight(
    tokens: List[str],
    activations: List[float],
    tokenizer: AutoTokenizer,
    max_idx: Optional[int] = None,
    min_max_act: Optional[float] = None,
    window_size: int = 50,
    show_full: bool = False,
) -> str:
    """
    Create HTML highlighting for tokens based on activation values.

    Args:
        tokens: List of token strings
        activations: List of activation values per token
        tokenizer: HuggingFace tokenizer
        max_idx: Index of maximum activation (auto-computed if None)
        min_max_act: Normalization value for activations
        window_size: Number of tokens to show around max activation
        show_full: Whether to show full sequence or windowed

    Returns:
        HTML string with highlighted tokens
    """
    act_tensor = torch.tensor(activations)

    if max_idx is None:
        max_idx = torch.argmax(act_tensor).item()

    # Apply windowing if not showing full sequence
    if not show_full:
        start_idx = max(0, max_idx - window_size)
        end_idx = min(len(tokens), max_idx + window_size + 1)
        tokens = tokens[start_idx:end_idx]
        act_tensor = act_tensor[start_idx:end_idx]

    return create_highlighted_tokens_html(
        tokens=tokens,
        activations=act_tensor,
        tokenizer=tokenizer,
        highlight_features=0,  # Single feature case
        color1=(255, 0, 0),  # Red color
        activation_names=["Activation"],
        min_max_act=min_max_act,
    )


def filter_examples_by_search(
    examples: List[Tuple[float, List[str], List[float], str]], search_term: str
) -> List[Tuple[float, List[str], List[float], str]]:
    """
    Filter examples by search term.

    Args:
        examples: List of (max_score, tokens, scores_per_token, text) tuples
        search_term: Term to search for in the text

    Returns:
        Filtered list of examples
    """
    if not search_term.strip():
        return examples

    search_term = search_term.lower().strip()
    filtered = []

    for max_score, tokens, scores_per_token, text in examples:
        if search_term in text.lower():
            filtered.append((max_score, tokens, scores_per_token, text))

    return filtered


@st.cache_data
def create_examples_html(
    examples: List[Tuple[float, List[str], List[float], str]],
    _tokenizer: AutoTokenizer,
    title: str = "Max Activating Examples",
    max_examples: int = 30,
    window_size: int = 50,
    use_absolute_max: bool = False,
    search_term: str = "",
) -> str:
    """
    Create HTML for a list of max activating examples.

    Args:
        examples: List of (max_score, tokens, scores_per_token, text) tuples
        _tokenizer: HuggingFace tokenizer
        title: Title for the HTML page
        max_examples: Maximum number of examples to display
        window_size: Number of tokens to show around max activation
        use_absolute_max: Whether to normalize using absolute maximum
        search_term: Optional search term to filter examples

    Returns:
        Complete HTML string
    """
    # Filter examples if search term provided
    if search_term.strip():
        examples = filter_examples_by_search(examples, search_term)

    content_parts = []
    min_max_act = None

    if use_absolute_max and examples:
        min_max_act = examples[0][0]  # First example has highest score

    # Only use the first 3 elements for backward compatibility
    for max_act, tokens, token_acts, text in examples[:max_examples]:
        max_idx = torch.argmax(torch.tensor(token_acts)).item()

        # Create both collapsed and full versions
        collapsed_html = create_html_highlight(
            tokens, token_acts, _tokenizer, max_idx, min_max_act, window_size, False
        )
        full_html = create_html_highlight(
            tokens, token_acts, _tokenizer, max_idx, min_max_act, window_size, True
        )

        content_parts.append(create_example_html(max_act, collapsed_html, full_html))

    return create_base_html(title=title, content=content_parts)


def render_streamlit_html(html_content: str, height: int = 800) -> None:
    """
    Render HTML content in Streamlit with proper styling.

    Args:
        html_content: HTML content to render
        height: Height of the component in pixels
    """
    # Use Streamlit's HTML component with scrolling
    st.components.v1.html(html_content, height=height, scrolling=True)


@st.fragment
def _render_statistics_tab(statistics_function: Callable, title: str):
    """Render statistics tab as a fragment to prevent full page reloads."""
    try:
        statistics_function()
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")


@st.fragment
def _render_interactive_tab(interactive_function: Callable, title: str):
    """Render interactive tab as a fragment to prevent full page reloads."""
    interactive_function()


def statistic_interactive_tab(
    statistics_function: Callable, interactive_function: Callable, title: str
):
    """
    Create a tab for statistics and interactive analysis with optimized rendering.
    Uses Streamlit fragments to prevent unnecessary full page reloads.

    Args:
        statistics_function: Function to compute statistics
        interactive_function: Function to create interactive analysis

    Returns:
        Streamlit tab component
    """
    st.subheader(title)

    tab1, tab2 = st.tabs(["📊 Dataset Statistics", "🔥 Interactive"])

    with tab1:
        _render_statistics_tab(statistics_function, title)

    with tab2:
        _render_interactive_tab(interactive_function, title)

    return tab1, tab2

