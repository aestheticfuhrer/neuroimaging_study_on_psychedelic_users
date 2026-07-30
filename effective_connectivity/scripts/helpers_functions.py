import numpy as np
import re

def extract_filename(path):
    # Split the path by '/' to get the last part (filename with extension)
    filename_with_extension = path.split('/')[-1]
    
    # Split the filename by '.' and take everything before the last part (extension)
    filename_without_extension = '.'.join(filename_with_extension.split('.')[:-1])
    
    return filename_without_extension

def extract_colname(colname, node_names):
    try:
        x = colname.split(node_names["sep"])
        y = node_names["skip"]
        return (x[y], x[y+1])
    except:
        return "error"
 
def contrast_effect(ranks, groups, control_group_name=None):
    """
    Fast calculation of effect size using pre-computed ranks.

    Parameters:
        ranks (np.ndarray): Pre-computed ranks
        groups (np.ndarray): Group assignments
        control_group_name: Name of control group (same type as groups)

    Returns:
        float: Cohen's d effect size on ranked data. 
            Specifically: Rosenthal's r = Z / sqrt(N), where Z is the normal approximation to the
            Mann-Whitney. Convert with:  d = 2r / sqrt(1 - r**2).
            Positive values indicate higher values in the non-control group.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(ranks)
    ranks, groups = ranks[valid_mask], groups[valid_mask]
    
    # Early exit if no data
    if len(ranks) == 0:
        return 0.0
    
    # Determine control group - no type conversion needed
    unique_groups = np.unique(groups)
    if control_group_name is not None and control_group_name in unique_groups:
        control_group = control_group_name
    else:
        control_group = unique_groups[0]
    
    # Direct comparison with original types
    control_mask = groups == control_group
    
    # Calculate effect size
    n1 = np.sum(control_mask)
    n2 = len(ranks) - n1
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    U = np.sum(ranks[control_mask]) - (n1 * (n1 + 1)) / 2
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if std_U < 1e-10:
        return 0.0

    Z = (mean_U - U) / std_U
    r = Z / np.sqrt(n1 + n2)   # Rosenthal's r, not Cohen's d

    return r

# Helper function to check if measurement passes the filter
def passes_filter(measurement_conditions, filter):
    if filter is None or measurement_conditions is None:
        return True  # No filter or no conditions means everything passes
    
    filter_type, filter_dict = filter
    
    for key, values in filter_dict.items():
        # Convert single values to lists for uniform processing
        if not isinstance(values, list):
            values = [values]
        
        # Check if the key exists in measurement conditions
        if key not in measurement_conditions:
            # For include: fail if key is missing
            # For exclude: pass if key is missing
            return filter_type == "exclude"
        
        # Check if the value matches
        value_matches = measurement_conditions[key] in values
        
        # For include: must match, for exclude: must not match
        if filter_type == "include" and not value_matches:
            return False
        if filter_type == "exclude" and value_matches:
            return False
    
    return True

# New helper functions

def calculate_ranks(values_array):
    """
    Calculate ranks for an array of values, properly handling ties.
    
    Parameters:
        values_array (np.ndarray): Array of values to rank
        
    Returns:
        np.ndarray: Array of ranks with ties averaged
    """
    # Filter out NaN values
    valid_mask = ~np.isnan(values_array)
    
    if not np.any(valid_mask):
        return np.full_like(values_array, np.nan)
        
    # Get valid values
    valid_values = values_array[valid_mask]
    
    # Calculate ranks for all valid values
    sorted_indices = np.argsort(valid_values)
    temp_ranks = np.zeros(len(valid_values))
    temp_ranks[sorted_indices] = np.arange(1, len(valid_values) + 1)
    
    # Handle ties
    unique_values, counts = np.unique(valid_values, return_counts=True)
    for j, count in enumerate(counts):
        if count > 1:  # It's a tie
            value = unique_values[j]
            tie_mask = valid_values == value
            temp_ranks[tie_mask] = np.mean(temp_ranks[tie_mask])
    
    # Create a full array with NaNs in invalid positions
    ranks = np.full_like(values_array, np.nan)
    ranks[valid_mask] = temp_ranks
    
    return ranks

def clean_parameter_name(param_name):
    """
    Clean up parameter names for display in regression results.
    
    Parameters:
        param_name (str): Original parameter name from regression model
        
    Returns:
        str: Cleaned parameter name for display
    """
    if param_name == "Intercept":
        return param_name
    
    # Handle complex parameter names with interactions
    clean_parts = []
    
    # Split by colon (interaction terms)
    parts = param_name.split(':')
    for part in parts:
        # Check if this part has a categorical term [T.value]
        cat_match = re.search(r'(.*?)\[T\.(.*?)\]', part)
        if cat_match:
            # Extract variable name and level
            var_name = cat_match.group(1)
            level = cat_match.group(2)
            clean_parts.append(level)
        else:
            # For non-intercept models, parameters might be prefixed with the variable name
            pure_match = re.search(r'^([a-zA-Z0-9_]+)$', part)
            if pure_match:
                # For main effects without categorical levels, just use the variable name
                clean_parts.append(pure_match.group(1))
            else:
                # Keep the original term if it doesn't match any pattern
                clean_parts.append(part)
    
    # Join with × for interactions
    if len(clean_parts) > 1:
        return ' × '.join(clean_parts)
    else:
        return clean_parts[0]

def format_significance_stars(p_value):
    """
    Return significance stars based on p-value.
    
    Parameters:
        p_value (float): P-value
        
    Returns:
        str: Significance stars (*, **, ***, or "")
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return ""
    
def calculate_confidence_interval(estimate, std_err, confidence=0.95, digits=3):
    """
    Calculate confidence interval based on standard error.
    
    Parameters:
        estimate (float): Parameter estimate
        std_err (float): Standard error
        confidence (float): Confidence level (default: 0.95)
        digits (int): Number of decimal places for formatting
    
    Returns:
        str: Formatted confidence interval string
    """
    # For 95% CI, z-score is approximately 1.96
    z = 1.96
    
    ci_lower = estimate - (std_err * z)
    ci_upper = estimate + (std_err * z)
    
    return f"[{ci_lower:.{digits}f}, {ci_upper:.{digits}f}]"


def get_model_info(model_summary, stat_name, default=None):
    """
    Safely extract model statistics from the summary DataFrame.
    
    Parameters:
        model_summary (pd.DataFrame): Summary DataFrame
        stat_name (str): Name of the statistic to extract
        default: Default value if statistic not found
        
    Returns:
        Value of the statistic or default if not found
    """
    if model_summary is None:
        return default
        
    try:
        return model_summary.loc[model_summary['Statistic'] == stat_name, 'Value'].values[0]
    except (IndexError, KeyError):
        return default


def format_parameter_row(param_name, estimate, std_err, t_value, p_value=None, 
                        ci_fmt=None, digits=3):
    """
    Format a parameter row for APA style output.
    
    Parameters:
        param_name (str): Parameter name
        estimate (float): Parameter estimate
        std_err (float): Standard error
        t_value (float): t-value
        p_value (float, optional): p-value
        ci_fmt (str, optional): Formatted confidence interval
        digits (int): Number of decimal places
        
    Returns:
        str: Formatted parameter row
    """
    from helpers_functions import clean_parameter_name, format_significance_stars
    
    # Clean parameter name
    clean_param_name = clean_parameter_name(param_name)
    
    # Format values with None checks
    estimate_fmt = f"{estimate:.{digits}f}" if estimate is not None else "N/A"
    std_err_fmt = f"{std_err:.{digits}f}" if std_err is not None else "N/A"
    t_value_fmt = f"{t_value:.{digits}f}" if t_value is not None else "N/A"
    
    # Start with basic info
    param_line = f"{clean_param_name:<20} {estimate_fmt:<10} {std_err_fmt:<10} {t_value_fmt:<10}"
    
    # Add p-value and significance stars if available
    if p_value is not None:
        p_value_fmt = f"{p_value:.{digits}f}"
        sig_stars = format_significance_stars(p_value)
        param_line += f" {p_value_fmt:<8}{sig_stars:<2}"
    
    # Add confidence interval if available
    if ci_fmt:
        param_line += f" {ci_fmt:<20}"
    
    return param_line


def get_permutation_se(param_name, param_estimates, perm_params, perm_t_values):
    """
    Calculate standard error from permutation data.
    
    Parameters:
        param_name (str): Parameter name
        param_estimates (pd.DataFrame): DataFrame of parameter estimates
        perm_params (dict): Dictionary of permuted parameter values
        perm_t_values (dict): Dictionary of permuted t-values
        
    Returns:
        float: Standard error from permutation data or None if not available
    """
    import numpy as np
    
    if param_name in perm_params and len(perm_params[param_name]) > 0:
        # Calculate standard deviation of parameter estimates from permutation data
        return np.std(perm_params[param_name])
    elif param_name in perm_t_values and len(perm_t_values[param_name]) > 1:
        # Get parameter estimate
        estimate = param_estimates.loc[param_estimates['Parameter'] == param_name, 'Estimate'].values[0]
        
        # Get t-values, excluding zeros to avoid division by zero
        t_values = np.array(perm_t_values[param_name])
        t_values = t_values[t_values != 0]
        
        if len(t_values) > 0:
            # Calculate SEs and then take the mean
            param_ses = np.abs(estimate / t_values)
            return np.mean(param_ses)
            
    return None

def process_single_permutation(keys, valid_measurements, samples, sample_to_keys, column, control_group_name):
    """
    Process a single permutation iteration
    
    Args:
        keys (list): Measurement keys
        valid_measurements (dict): Validated measurement data
        samples (dict): Sample dataframes
        sample_to_keys (dict): Mapping of sample IDs to keys
        column (str): Column to permute
        control_group_name (str): Name of control group
    
    Returns:
        dict: Permuted contrast results for this iteration
    """
    perm_result = {}
    sample_permutations = {}

    # Shuffle samples
    for sample_id in sample_to_keys.keys():
        if sample_id in samples:
            sample_df = samples[sample_id]
            
            if column in sample_df.columns:
                indices = np.arange(len(sample_df))
                permuted_indices = np.random.permutation(indices)
                
                id_to_permuted_value = {
                    id_val: sample_df[column].values[permuted_indices[i]] 
                    for i, id_val in enumerate(sample_df['ID'].values)
                }
                
                sample_permutations[sample_id] = id_to_permuted_value

    # Apply permutations to measurements
    for key in keys:
        sample_id = valid_measurements[key]["sample_id"]
        
        if sample_id in sample_permutations:
            measurement_ids = [str(id_val) for id_val in valid_measurements[key]["id"]]
            ranks = valid_measurements[key]["ranks"]
            
            # Get permuted column values
            permuted_column_values = [
                sample_permutations[sample_id].get(id_val, 
                    samples[sample_id].loc[samples[sample_id]['ID'] == id_val, column].iloc[0] 
                    if not samples[sample_id].loc[samples[sample_id]['ID'] == id_val].empty 
                    else "unknown"
                ) for id_val in measurement_ids
            ]
            
            permuted_column = np.array(permuted_column_values)
            
            # Calculate contrast
            perm_result[key] = contrast_effect(ranks, permuted_column, control_group_name)
        else:
            # Fallback permutation logic
            if "group" in valid_measurements[key]:
                original_column = np.array(valid_measurements[key]["group"])
                permuted_indices = np.random.permutation(len(original_column))
                permuted_column = original_column[permuted_indices]
                perm_result[key] = contrast_effect(
                    valid_measurements[key]["ranks"], 
                    permuted_column, 
                    control_group_name
                )

    return perm_result


def permute_worker(iteration_seed, measurement_cache, sample_lookup, control_group_name):
    """
    Worker function for single permutation iteration - designed for parallel execution.

    This function performs one complete permutation cycle:
    1. Sets random seed for reproducibility
    2. Shuffles group assignments within each sample
    3. Recalculates contrast effects for all measurements
    4. Returns results with summary statistics

    Parameters:
        iteration_seed (int or None): Random seed for this iteration
        measurement_cache (dict): Pre-processed measurement data
            Keys: measurement names
            Values: dict with 'valid_ranks', 'valid_ids', 'sample_id'
        sample_lookup (dict): Sample ID -> {'ids': array, 'values': array}
        control_group_name (str): Name of control group for contrast calculation

    Returns:
        tuple: (perm_result, mean_contrast, median_contrast)
            - perm_result (dict): measurement_name -> contrast_effect
            - mean_contrast (float): Mean of all contrasts in this permutation
            - median_contrast (float): Median of all contrasts in this permutation
    """
    # Set seed for this iteration
    if iteration_seed is not None:
        np.random.seed(iteration_seed)

    # Generate permutations for each sample
    sample_permutations = {}
    for sample_id, data in sample_lookup.items():
        indices = np.random.permutation(len(data['ids']))
        sample_permutations[sample_id] = dict(zip(data['ids'], data['values'][indices]))

    # Process measurements using cached data
    perm_result = {}
    for key, cache in measurement_cache.items():
        sample_id = cache['sample_id']
        valid_ranks = cache['valid_ranks']
        valid_ids = cache['valid_ids']

        # Get permuted values for valid IDs
        permuted_values = np.array([sample_permutations[sample_id].get(id_val, None)
                                for id_val in valid_ids])

        # Remove any None values that couldn't be mapped
        good_indices = permuted_values != None
        if not np.any(good_indices):
            continue

        # Calculate contrast with valid data only
        perm_result[key] = contrast_effect(
            valid_ranks[good_indices],
            permuted_values[good_indices],
            control_group_name
        )

    # Calculate mean and median for this permutation
    if perm_result:
        contrast_values = list(perm_result.values())
        mean_contrast = np.mean(contrast_values)
        median_contrast = np.median(contrast_values)
    else:
        mean_contrast = np.nan
        median_contrast = np.nan

    return (perm_result, mean_contrast, median_contrast)


def permute_worker_batch(iteration_seeds, measurement_cache, sample_lookup, control_group_name):
    """
    BATCHED worker - processes multiple permutations to reduce serialization overhead.

    This is MUCH faster for parallel execution as it amortizes the cost of
    pickling/unpickling large data structures across many permutations.

    Parameters:
        iteration_seeds (list): List of random seeds, one per permutation in this batch
        measurement_cache (dict): Pre-processed measurement data
        sample_lookup (dict): Sample ID -> {'ids': array, 'values': array}
        control_group_name (str): Name of control group

    Returns:
        list: List of (perm_result, mean_contrast, median_contrast) tuples
    """
    results = []
    for seed in iteration_seeds:
        results.append(permute_worker(seed, measurement_cache, sample_lookup, control_group_name))
    return results