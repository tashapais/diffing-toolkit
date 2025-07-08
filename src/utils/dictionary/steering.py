from typing import List, Dict, Any
import pandas as pd
import torch
from nnsight import LanguageModel
import pandas as pd
from collections import defaultdict
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from tiny_dashboard.utils import apply_chat
from src.utils.dictionary.utils import load_latent_df, load_dictionary_model

def load_prompts(prompts_file: str) -> List[str]:
    with open(prompts_file, 'r') as f:
        prompts = f.readlines()
    return prompts

def get_sae_latent(latent_idx: int, dictionary_model) -> torch.Tensor:
    dict_size = dictionary_model.dict_size
    assert 0 <= latent_idx < dict_size, f"Latent index {latent_idx} out of range [0, {dict_size})"

    decoder_vector = dictionary_model.decoder.weight[:, latent_idx]  # [activation_dim]
    return decoder_vector.detach()

def run_latent_steering_experiment(
    method,
    get_latent_fn,
    dictionary_name: str,
    layer: int,
    results_dir: Path,
):
    latent_steering_cfg = method.cfg.diffing.method.analysis.latent_steering

    results_dir = results_dir / "latent_steering"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"{latent_steering_cfg.prompts_file.split('/')[-1].split('.')[0]}.csv"
    if not latent_steering_cfg.overwrite and results_file.exists():
        logger.info(f"Skipping latent steering experiment because results already exist at {results_dir}")
        return
    
    prompts_file = latent_steering_cfg.prompts_file
    latent_df = load_latent_df(dictionary_name)
    target_column = latent_steering_cfg.target_column
    k = latent_steering_cfg.k
    largest = latent_steering_cfg.largest
    
    prompts = load_prompts(prompts_file)


    dictionary_model = load_dictionary_model(dictionary_name)

    results = latent_steering_experiment(
        get_latent_fn=lambda latent_idx: get_latent_fn(latent_idx, dictionary_model=dictionary_model),
        prompts=prompts,
        latent_df=latent_df,
        target_column=target_column,
        k=k,
        base_model=method.base_model,
        finetuned_model=method.finetuned_model,
        tokenizer=method.tokenizer,
        layer=layer,
        largest=largest,
        batch_size=48,
        max_length=latent_steering_cfg.max_length,
        temperature=latent_steering_cfg.temperature,
        do_sample=latent_steering_cfg.do_sample,
        device=latent_steering_cfg.device,
        use_chat_formatting=latent_steering_cfg.use_chat_formatting,
        enable_thinking=latent_steering_cfg.enable_thinking,
        steering_factors_percentages=latent_steering_cfg.steering_factors_percentages,
        steering_modes=latent_steering_cfg.steering_modes,
    )

    save_results_to_csv(results, results_file)
    return results
    

def latent_steering_experiment(
    get_latent_fn,             # Function: latent_idx -> latent_vector  
    prompts: List[str],        # List of prompts to test
    latent_df: pd.DataFrame,   # DataFrame with latent statistics
    target_column: str,        # Column to sort by for top-k selection
    k: int,                    # Number of top latents to test
    base_model,                # Base language model
    finetuned_model,           # Finetuned language model
    tokenizer,                 # Tokenizer
    layer: int,                # Layer to apply steering to
    largest: bool = False,      # Whether to use largest latents
    batch_size: int = 8,       # Batch size for parallel generation
    max_length: int = 50,      # Max tokens to generate
    temperature: float = 1.0,  # Generation temperature
    do_sample: bool = True,    # Whether to use sampling
    device: str = "cuda",      # Device for computation
    use_chat_formatting: bool = True,  # Whether to apply chat formatting
    enable_thinking: bool = False,     # Whether to enable thinking,
    steering_factors_percentages: List[float] = [0.5, 0.8, 1.0, 1.5],
    steering_modes: List[str] = ["all_tokens", "prompt_only"],
) -> List[Dict[str, Any]]:
    """
    Run batched steering experiments for both base and finetuned models.
    
    For each prompt, generates text using different latent steering configurations
    in batches for efficiency. Tests both base and finetuned models.
    Each batch maintains a consistent steering mode for clean nnsight logic.
    
    Args:
        get_latent_fn: Function that takes latent_idx and returns latent vector
        prompts: List of prompts to test
        latent_df: DataFrame with latent statistics including target_column
        target_column: Column name to sort latents by for top-k selection
        k: Number of top latents to test
        base_model: Base language model
        finetuned_model: Finetuned language model  
        tokenizer: Tokenizer for both models
        layer: Layer index to apply steering to
        largest: Whether to use largest latents (default: False for smallest)
        batch_size: Number of steering configs to process in parallel per batch
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling vs greedy
        device: Device for computation
        use_chat_formatting: Whether to apply chat template to prompts
        enable_thinking: Whether to enable thinking
        steering_factors_percentages: List of steering factors percentages to test
        steering_modes: List of steering modes to test

    Returns:
        List of result dictionaries containing all generation results
    """

    
    # Validate inputs
    assert len(prompts) > 0, "Must provide at least one prompt"
    assert k > 0, "k must be positive"
    assert batch_size > 0, "batch_size must be positive"  
    assert target_column in latent_df.columns, f"target_column '{target_column}' not found in latent_df"
    assert layer >= 0, "layer must be non-negative"
    
    # Model selection
    models = {
        'base': base_model,
        'finetuned': finetuned_model
    }
    
    # Top-K latent selection
    if largest:
        top_latents = latent_df.nlargest(k, target_column)
    else:
        top_latents = latent_df.nsmallest(k, target_column)
    assert len(top_latents) == k, f"Only {len(top_latents)} latents available, requested {k}"
    
    
    # Create all steering configurations
    all_steering_configs = []
    for idx, row in top_latents.iterrows():
        # Try validation first, then train, then fail
        if 'max_act_validation' in row:
            max_act = row['max_act_validation']
        elif 'max_act_train' in row:
            max_act = row['max_act_train']
        else:
            raise KeyError(f"Neither 'max_act_validation' nor 'max_act_train' found for latent {idx}")
            
        for percentage in steering_factors_percentages:
            for mode in steering_modes:
                config = {
                    'latent_idx': idx,
                    'steering_factor': percentage * max_act,
                    'steering_factor_percentage': percentage,
                    'steering_mode': mode,
                    'max_act': max_act,
                    'target_value': row[target_column]
                }
                all_steering_configs.append(config)
    
    logger.info(f"Created {len(all_steering_configs)} steering configurations for {k} top latents")
    
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Processing {model_name} model...")
        
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"  Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
            # Create batches for this prompt, grouped by steering mode
            batches = _create_batches_for_prompt_by_mode(prompt, all_steering_configs, batch_size, use_chat_formatting, tokenizer, enable_thinking)
            
            for batch_idx, batch_data in tqdm(enumerate(batches), desc=f"Processing batches for prompt {prompt_idx + 1}", total=len(batches)):
                # Generate for this batch
                generated_texts = _generate_with_steering_batched_single_mode(
                    model=model,
                    tokenizer=tokenizer,
                    batch_data=batch_data,
                    get_latent_fn=get_latent_fn,
                    layer=layer,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    device=device
                )
                
                # Store results
                for config, generated_text in zip(batch_data['configs'], generated_texts):
                    result = {
                        'model_type': model_name,
                        'latent_idx': config['latent_idx'],
                        'steering_factor': config['steering_factor'],
                        'steering_factor_percentage': config['steering_factor_percentage'],
                        'steering_mode': config['steering_mode'],
                        'max_act': config['max_act'],
                        'target_value': config['target_value'],
                        'is_baseline': config.get('is_baseline', False),
                        'prompt': prompt,
                        'formatted_prompt': batch_data['formatted_prompt'],
                        'generated_text': generated_text,
                        'layer': layer,
                        'batch_idx': batch_idx,
                        'prompt_idx': prompt_idx
                    }
                    results.append(result)
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    
    logger.info(f"Completed experiment. Generated {len(results)} total results.")
    return results


def _create_batches_for_prompt_by_mode(prompt: str, steering_configs: List[Dict], batch_size: int, use_chat_formatting: bool, tokenizer, enable_thinking: bool) -> List[Dict]:
    """
    Create batches for a single prompt with different steering configurations.
    Groups configurations by steering mode to ensure consistent mode per batch.
    """
    
    # Add baseline (no steering) as first config
    baseline_config = {
        'latent_idx': None,
        'steering_factor': 0.0,
        'steering_factor_percentage': 0.0,
        'steering_mode': 'baseline',
        'max_act': 0.0,
        'target_value': 0.0,
        'is_baseline': True
    }
    
    # Format prompt once
    if use_chat_formatting:
        formatted_prompt = apply_chat(prompt, tokenizer, add_bos=False, enable_thinking=enable_thinking)
    else:
        formatted_prompt = prompt
    
    # Group configurations by steering mode
    configs_by_mode = defaultdict(list)
    configs_by_mode['baseline'] = [baseline_config]
    
    for config in steering_configs:
        mode = config['steering_mode']
        configs_by_mode[mode].append(config)
    
    # Create batches for each mode
    batches = []
    for mode, mode_configs in configs_by_mode.items():
        # Create batches within this mode
        for i in range(0, len(mode_configs), batch_size):
            batch_configs = mode_configs[i:i+batch_size]
            
            batch = {
                'prompt': prompt,
                'formatted_prompt': formatted_prompt,
                'configs': batch_configs,
                'batch_size': len(batch_configs),
                'steering_mode': mode  # All configs in this batch have the same mode
            }
            batches.append(batch)
    
    return batches


def _generate_with_steering_batched_single_mode(
    model, 
    tokenizer, 
    batch_data: Dict,
    get_latent_fn,
    layer: int,
    max_length: int,
    temperature: float,
    do_sample: bool,
    device: str
) -> List[str]:
    """
    Generate text for a batch of steering configurations with a single steering mode.
    This simplifies the nnsight logic since all batch elements use the same steering mode.
    """
        
    formatted_prompt = batch_data['formatted_prompt']
    configs = batch_data['configs']
    actual_batch_size = batch_data['batch_size']
    steering_mode = batch_data['steering_mode']
    
    # Tokenize prompt once
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    
    # Shape assertions
    assert input_ids.ndim == 2, f"Expected 2D input_ids, got shape {input_ids.shape}"
    assert input_ids.shape[0] == 1, f"Expected batch size 1 from tokenizer, got {input_ids.shape[0]}"
    
    # Repeat input_ids for batch
    batch_input_ids = input_ids.repeat(actual_batch_size, 1)  # [batch_size, seq_len]
    assert batch_input_ids.shape == (actual_batch_size, input_ids.shape[1]), f"Unexpected batch shape: {batch_input_ids.shape}"
    
    # Prepare steering vectors and factors for the batch
    steering_vectors = []
    steering_factors = []
    
    hidden_size = model.config.hidden_size
    
    for config in configs:
        if config.get('is_baseline', False):
            # No steering for baseline
            steering_vectors.append(torch.zeros(hidden_size, device=device))
            steering_factors.append(0.0)
        else:
            latent_vector = get_latent_fn(config['latent_idx'])
            assert latent_vector.shape == (hidden_size,), f"Expected latent vector shape ({hidden_size},), got {latent_vector.shape}"
            steering_vectors.append(latent_vector.to(device))
            steering_factors.append(config['steering_factor'])
    
    # Stack steering vectors: [batch_size, hidden_dim]
    steering_vectors_batch = torch.stack(steering_vectors)
    steering_factors_tensor = torch.tensor(steering_factors, device=device)  # [batch_size]
    
    assert steering_vectors_batch.shape == (actual_batch_size, hidden_size), f"Unexpected steering batch shape: {steering_vectors_batch.shape}"
    assert steering_factors_tensor.shape == (actual_batch_size,), f"Unexpected steering factors shape: {steering_factors_tensor.shape}"
    
    # Create LanguageModel wrapper
    nn_model = LanguageModel(model, tokenizer=tokenizer)
    
    # Generate with consistent steering mode for entire batch
    with nn_model.generate(
        batch_input_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        disable_compile=True,
    ) as tracer:
        
        if steering_mode == "baseline":
            # No steering applied - generate normally
            pass
            
        elif steering_mode == "all_tokens":
            # Apply steering to all tokens for the entire batch
            with nn_model.model.layers[layer].all():
                # Broadcast steering: [batch_size, hidden_dim] * [batch_size, 1] -> [batch_size, hidden_dim]
                steering_additive = steering_vectors_batch * steering_factors_tensor.unsqueeze(1)
                nn_model.model.layers[layer].output[0][:] += steering_additive.unsqueeze(1)
                
        elif steering_mode == "prompt_only":
            # Apply steering only during prompt processing for the entire batch
            steering_additive = steering_vectors_batch * steering_factors_tensor.unsqueeze(1)
            nn_model.model.layers[layer].output[0][:] += steering_additive.unsqueeze(1)
            
            # Move to next tokens without applying steering
            for i in range(max_length):
                nn_model.model.layers[layer].next()
                
        else:
            raise ValueError(f"Unknown steering mode: {steering_mode}")
        
        # Save the output
        outputs = nn_model.generator.output.save()
    
    # Shape assertion for outputs
    assert outputs.shape[0] == actual_batch_size, f"Expected {actual_batch_size} outputs, got {outputs.shape[0]}"
    
    # Decode all generated texts
    generated_texts = []
    for i in range(actual_batch_size):
        text = tokenizer.decode(outputs[i], skip_special_tokens=False)
        generated_texts.append(text)
    
    assert len(generated_texts) == actual_batch_size, f"Expected {actual_batch_size} generated texts, got {len(generated_texts)}"
    
    return generated_texts


def save_results_to_csv(results: List[Dict], filepath: str) -> None:
    """
    Save results to CSV file for analysis.
    """
    assert len(results) > 0, "No results to save"
    
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(results)} results to {filepath}")


def display_steering_results(results_dir: Path, dictionary_name: str) -> None:
    """
    Display steering experiment results in Streamlit interface.
    
    Args:
        results_dir: Directory containing steering results CSV files
        dictionary_name: Name of the dictionary/SAE model
    """
    import streamlit as st
    from pathlib import Path
    
    st.markdown("### Latent Steering Experiment Results")
    
    # Find steering results directory
    steering_dir = results_dir / "latent_steering"
    if not steering_dir.exists():
        st.error(f"No steering results found at {steering_dir}")
        st.info("Run the latent steering experiment first to generate results.")
        return
    
    # Find CSV files in the steering directory
    csv_files = list(steering_dir.glob("*.csv"))
    if not csv_files:
        st.error(f"No CSV result files found in {steering_dir}")
        return
    
    # Let user select which results file to view if multiple exist
    if len(csv_files) > 1:
        selected_file = st.selectbox(
            "Select Results File:",
            options=csv_files,
            format_func=lambda x: x.name,
            help="Choose which steering experiment results to display"
        )
    else:
        selected_file = csv_files[0]
        st.info(f"Displaying results from: {selected_file.name}")
    
    # Load results
    try:
        df = pd.read_csv(selected_file)
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return
    
    if len(df) == 0:
        st.warning("No results found in the selected file.")
        return
    
    # Get unique latents for selection
    unique_latents = sorted([idx for idx in df['latent_idx'].unique() if pd.notna(idx)])
    if not unique_latents:
        st.error("No valid latent indices found in results.")
        return
    
    # Latent selector at the top
    st.markdown("#### Select Latent to Analyze")
    selected_latent = st.selectbox(
        "Latent Index:",
        options=unique_latents,
        help="Choose which latent's steering effects to display"
    )
    
    # Show target value for selected latent
    latent_target_value = df[df['latent_idx'] == selected_latent]['target_value'].iloc[0]
    st.caption(f"Target Value (Index): {int(latent_target_value)}")
    
    # Filter results for selected latent (including baseline)
    latent_results = df[(df['latent_idx'] == selected_latent) | (df['is_baseline'] == True)]
    
    if len(latent_results) == 0:
        st.warning(f"No results found for latent {selected_latent}")
        return
    
    # Get available filter options from the data
    available_modes = sorted(latent_results[latent_results['is_baseline'] == False]['steering_mode'].unique())
    available_factors = sorted(latent_results[latent_results['is_baseline'] == False]['steering_factor_percentage'].unique())
    
    # Filtering controls
    st.markdown("#### Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model selector
        available_models = sorted(latent_results['model_type'].unique())
        selected_model = st.selectbox(
            "Model:",
            options=available_models,
            help="Choose which model results to display"
        )
    
    with col2:
        selected_modes = st.multiselect(
            "Steering Modes:",
            options=available_modes,
            default=available_modes,
            help="Choose which steering modes to display"
        )
    
    with col3:
        selected_factors = st.multiselect(
            "Steering Strengths:",
            options=available_factors,
            default=available_factors,
            format_func=lambda x: f"{x}×",
            help="Choose which steering strength factors to display"
        )
    
    # Filter results based on selections
    if selected_modes and selected_factors:
        filtered_results = latent_results[
            # Include baseline for selected model
            ((latent_results['is_baseline'] == True) & (latent_results['model_type'] == selected_model)) |
            # Include steered results for selected model that match other filters
            ((latent_results['model_type'] == selected_model) & 
             (latent_results['steering_mode'].isin(selected_modes)) & 
             (latent_results['steering_factor_percentage'].isin(selected_factors)))
        ]
    else:
        # If no steering filters selected, show only baseline for selected model
        filtered_results = latent_results[
            (latent_results['is_baseline'] == True) & (latent_results['model_type'] == selected_model)
        ]
    
    if len(filtered_results) == 0:
        st.warning("No results match the selected filters.")
        return
    
    # Group by prompt and model
    grouped_results = _group_results_for_display(filtered_results)
    
    # Add comprehensive copy button for all filtered results
    st.markdown("#### Export All Results")
    all_results_text = _format_all_results_for_llm_analysis(selected_latent, int(latent_target_value), grouped_results, selected_modes, selected_factors, selected_model)
    
    st.download_button(
        label="📋 Copy All Results for LLM Analysis",
        data=all_results_text,
        file_name=f"steering_analysis_latent{selected_latent}_{selected_model}_all_prompts.txt",
        mime="text/plain",
        help="Download formatted text containing all filtered results for LLM interpretability analysis",
        use_container_width=True
    )
    
    # Display results for each prompt
    st.markdown("#### Steering Results by Prompt")
    
    for group_key, group_data in grouped_results.items():
        model_type, prompt_idx = group_key.split('_prompt')
        prompt_text = group_data['baseline']['prompt']
        
        # Create expandable section for each prompt
        with st.expander(f"**{model_type.title()} Model - Prompt {int(prompt_idx) + 1}:** {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}", expanded=False):
            
            # Show baseline first
            baseline = group_data['baseline']
            st.markdown("##### 🔹 Baseline (No Steering)")
            st.code(baseline['generated_text'], language="text", wrap_lines=True)
            
            # Show steered results
            if group_data['steered']:
                st.markdown("##### 🎯 Steered Results")
                
                # Group steered results by steering mode and factor
                steered_by_config = defaultdict(list)
                for result in group_data['steered']:
                    key = (result['steering_mode'], result['steering_factor_percentage'])
                    steered_by_config[key].append(result)
                
                # Display in organized way
                for (mode, factor_pct), mode_results in steered_by_config.items():
                    st.markdown(f"**Mode: {mode}, Factor: {factor_pct}×**")
                    
                    for result in mode_results:
                        # Compact stats at the top (only steering factor)
                        st.caption(f"🎯 Steering Factor: {result['steering_factor']:.2f}")
                        
                        # Full-width generated text
                        st.code(result['generated_text'], language="text", wrap_lines=True)
                        
                        st.markdown("---")
            else:
                st.info("No steered results found for this configuration.")


def _group_results_for_display(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Group results by model and prompt for display purposes.
    
    Returns:
        Dict with keys like "base_prompt0", "finetuned_prompt0" etc.,
        each containing {'baseline': result_dict, 'steered': [result_dict, ...]}
    """
    grouped = defaultdict(lambda: {'baseline': None, 'steered': []})
    
    for _, result in df.iterrows():
        result_dict = result.to_dict()
        key = f"{result['model_type']}_prompt{result['prompt_idx']}"
        
        if result['is_baseline']:
            grouped[key]['baseline'] = result_dict
        else:
            grouped[key]['steered'].append(result_dict)
    
    # Validate that every group has a baseline
    for key, group in grouped.items():
        assert group['baseline'] is not None, f"No baseline found for {key}"
    
    return dict(grouped)


def _format_results_for_llm_analysis(latent_idx: int, group_data: Dict, model_type: str, prompt_idx: str) -> str:
    """
    Format steering results in a structured format suitable for LLM interpretability analysis.
    
    Args:
        latent_idx: The latent index being analyzed
        group_data: Dictionary containing baseline and steered results
        model_type: 'base' or 'finetuned'
        prompt_idx: Prompt index as string
        
    Returns:
        Formatted text string for LLM analysis
    """
    baseline = group_data['baseline']
    steered_results = group_data['steered']
    
    # Header section
    output = []
    output.append("=" * 80)
    output.append(f"LATENT STEERING ANALYSIS - Latent {latent_idx}")
    output.append("=" * 80)
    output.append("")
    
    # Metadata
    output.append(f"Model: {model_type.title()}")
    output.append(f"Prompt Index: {prompt_idx}")
    output.append(f"Latent Index: {latent_idx}")
    
    # Add latent metadata if available from steered results
    if steered_results:
        first_steered = steered_results[0]
        output.append(f"Max Activation: {first_steered['max_act']:.4f}")
        output.append(f"Target Value: {first_steered['target_value']:.4f}")
        output.append(f"Layer: {first_steered['layer']}")
    
    output.append("")
    
    # Original prompt
    output.append("ORIGINAL PROMPT:")
    output.append("-" * 40)
    output.append(baseline['prompt'])
    output.append("")
    
    # Baseline generation
    output.append("BASELINE GENERATION (No Steering):")
    output.append("-" * 40)
    output.append(baseline['generated_text'])
    output.append("")
    
    # Steered results
    if steered_results:
        output.append("STEERED GENERATIONS:")
        output.append("-" * 40)
        
        # Group by steering configuration
        steered_by_config = defaultdict(list)
        for result in steered_results:
            key = (result['steering_mode'], result['steering_factor_percentage'])
            steered_by_config[key].append(result)
        
        # Sort by mode and factor for consistent ordering
        sorted_configs = sorted(steered_by_config.items(), key=lambda x: (x[0][0], x[0][1]))
        
        for (mode, factor_pct), mode_results in sorted_configs:
            for result in mode_results:
                output.append("")
                output.append(f"[Mode: {mode}, Factor: {factor_pct}x, Actual Strength: {result['steering_factor']:.4f}]")
                output.append(result['generated_text'])
    else:
        output.append("STEERED GENERATIONS:")
        output.append("-" * 40)
        output.append("No steered results available for this configuration.")
    
    output.append("")
    output.append("=" * 80)
    output.append("END OF ANALYSIS")
    output.append("=" * 80)
    
    return "\n".join(output)


def _format_all_results_for_llm_analysis(latent_idx: int, target_value_idx: int, all_grouped_results: Dict, selected_modes: List[str], selected_factors: List[float], selected_model: str) -> str:
    """
    Format all steering results into one comprehensive text for LLM analysis.
    
    Args:
        latent_idx: The latent index being analyzed
        target_value_idx: Target value as integer index
        all_grouped_results: All grouped results across prompts and models
        selected_modes: Selected steering modes
        selected_factors: Selected steering factor percentages
        selected_model: Selected model type
        
    Returns:
        Comprehensive formatted text string for LLM analysis
    """
    output = []
    output.append("=" * 100)
    output.append(f"COMPREHENSIVE LATENT STEERING ANALYSIS - Latent {latent_idx}")
    output.append("=" * 100)
    output.append("")
    
    # Overall metadata
    output.append(f"Latent Index: {latent_idx}")
    output.append(f"Target Value (Index): {target_value_idx}")
    output.append(f"Model: {selected_model.title()}")
    output.append(f"Analyzed Steering Modes: {', '.join(selected_modes)}")
    output.append(f"Analyzed Steering Strengths: {', '.join([f'{f}×' for f in selected_factors])}")
    output.append("")
    
    # Process all results
    for group_key, group_data in all_grouped_results.items():
        model_type, prompt_idx = group_key.split('_prompt')
        baseline = group_data['baseline']
        steered_results = group_data['steered']
        
        output.append("=" * 80)
        output.append(f"PROMPT {int(prompt_idx) + 1}")
        output.append("=" * 80)
        output.append("")
        
        # Original prompt
        output.append("PROMPT:")
        output.append(baseline['prompt'])
        output.append("")
        
        # Baseline
        output.append("BASELINE (No Steering):")
        output.append(baseline['generated_text'])
        output.append("")
        
        # Steered results
        if steered_results:
            output.append("STEERED GENERATIONS:")
            output.append("-" * 50)
            
            # Group by steering configuration
            steered_by_config = defaultdict(list)
            for result in steered_results:
                key = (result['steering_mode'], result['steering_factor_percentage'])
                steered_by_config[key].append(result)
            
            # Sort by mode and factor for consistent ordering
            sorted_configs = sorted(steered_by_config.items(), key=lambda x: (x[0][0], x[0][1]))
            
            for (mode, factor_pct), mode_results in sorted_configs:
                for result in mode_results:
                    output.append(f"[Mode: {mode}, Strength: {factor_pct}× = {result['steering_factor']:.2f}]")
                    output.append(result['generated_text'])
                    output.append("")
        else:
            output.append("STEERED GENERATIONS:")
            output.append("No steered results for selected filters.")
        
        output.append("")
    
    output.append("=" * 100)
    output.append("END OF COMPREHENSIVE ANALYSIS")
    output.append("=" * 100)
    
    return "\n".join(output)
