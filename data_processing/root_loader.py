
import uproot
import pandas as pd
import numpy as np
import awkward as ak
import os

def load_root_file(file_path, tree_name="events", branches=None, library="pd"):
    """
    Loads a ROOT file.
    
    Args:
        file_path (str): Path to the .root file.
        tree_name (str): Name of the tree to load.
        branches (list): List of branches to load. If None, loads all.
        library (str): "pd" for Pandas DataFrame, "ak" for Awkward Array.
    
    Returns:
        pd.DataFrame or ak.Array: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with uproot.open(file_path) as file:
            if tree_name not in file:
                raise ValueError(f"Tree '{tree_name}' not found in {file_path}. Available keys: {file.keys()}")
            
            tree = file[tree_name]
            
            if library == "ak":
                return tree.arrays(branches, library="ak")
            elif library == "pd":
                # For nested data, uproot's library="pd" might explode data or behave differently
                # depending on structure.
                # Using awkward to pandas conversion is often more controlled.
                arrays = tree.arrays(branches, library="ak")
                # ak.to_dataframe might create a MultiIndex if nested.
                # If we want to keep lists in cells (like the original code seems to imply with get_single_event_df),
                # we might accept the simple conversion or process it.
                # However, standard pandas doesn't support list-columns well for all ops.
                # But looking at utilities.py: ligne["hitx"] is used as a list.
                # So we want a DF where cells contain arrays/lists.
                
                # Iterate and convert to dict of lists/arrays is one way, or just let uproot handle it.
                # uproot.tree.arrays(library="pd") flattens by default? No.
                return tree.arrays(branches, library="pd") 
                
    except Exception as e:
        print(f"Error loading ROOT file: {e}")
        return None

def compute_event_level_features(df):
    """
    Computes event-level features like charge_totale, n_hits, etc.
    Assumes columns like 'charge', 'time' contain lists/arrays of hit data.
    """
    # Create copies to avoid SettingWithCopy warnings if df is a slice
    df = df.copy()
    
    if 'charge' in df.columns:
        # Check if first element is list-like
        if df.shape[0] > 0 and isinstance(df['charge'].iloc[0], (list, np.ndarray, ak.Array)):
             df['charge_totale'] = df['charge'].apply(np.sum)
             df['n_hits'] = df['charge'].apply(len)
             df['max_charge'] = df['charge'].apply(lambda x: np.max(x) if len(x) > 0 else 0)
             df['min_charge'] = df['charge'].apply(lambda x: np.min(x) if len(x) > 0 else 0)
    
    return df
