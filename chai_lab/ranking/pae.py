import numpy as np


def calculate_chain_pair_pae_min(pae: np.ndarray, chain_lengths: list[int]) -> dict:
    """
    Calculate the minimum PAE for the binder chain and each target chain.

    Args:
    pae (np.ndarray): The full PAE matrix, shape (num_residues, num_residues)
    chain_lengths (list[int]): List of lengths for each chain

    Returns:
    dict: Dictionary containing:
        - binder_pae_min: minimum PAE for the binder chain
        - target_pae_mins: array of minimum PAE values for each target chain
        - target_global_min: minimum PAE across all target chains
    """
    num_chains = len(chain_lengths)
    chain_pae_min = np.zeros(num_chains)
    
    # Calculate the starting index for each chain
    chain_starts = [0] + list(np.cumsum(chain_lengths[:-1]))
    
    # Calculate minimum PAE for each chain
    for i in range(num_chains):
        # Get the submatrix for this chain
        submatrix = pae[chain_starts[i]:chain_starts[i]+chain_lengths[i],
                       chain_starts[i]:chain_starts[i]+chain_lengths[i]]
        
        # Calculate the minimum PAE for this chain
        chain_pae_min[i] = np.min(submatrix)
    
    # Get binder (first chain) and target minimums
    binder_pae_min = chain_pae_min[0]  # First chain is the binder
    target_pae_mins = chain_pae_min[1:]  # All other chains are targets
    
    # Calculate global minimum for all targets
    target_pae_global_min = np.min(target_pae_mins)
    
    return binder_pae_min, target_pae_mins, target_pae_global_min

#! OLD
# def calculate_pae_interaction(pae: np.ndarray, binder_length: int) -> tuple[float, float, float]:
#     """
#     Calculate the mean PAE values for the interaction between the binder chain and each target chain.

#     Args:
#     pae (np.ndarray): The full PAE matrix, shape (num_residues, num_residues)
#     binder_length (int): The length of the binder chain

#     Returns:
#     tuple[float, float, float]: A tuple containing:
#         - binder_pae_interaction_mean: mean PAE value for the interaction between the binder chain and each target chain
#         - target_pae_interaction_mean: mean PAE value for the interaction between each target chain and the binder chain
#         - pae_interaction_total: total PAE value for the interaction between the binder chain and each target chain
#     """

#     pae_interaction1 = np.mean(pae[:binder_length, binder_length:])
#     pae_interaction2 = np.mean(pae[binder_length:, :binder_length])
#     binder_pae_interaction_mean = np.mean(pae[:binder_length, :binder_length])
#     target_pae_interaction_mean = np.mean(pae[binder_length:, binder_length:])
#     pae_interaction_total = (pae_interaction1 + pae_interaction2) / 2

#     return binder_pae_interaction_mean, target_pae_interaction_mean, pae_interaction_total


def calculate_pae_interaction(pae: np.ndarray, chain_lengths: list[int]) -> tuple[float, float, float, list[float]]:
    """
    Calculate the mean PAE values for the interaction between the binder chain and each target chain.

    Args:
    pae (np.ndarray): The full PAE matrix, shape (num_residues, num_residues)
    chain_lengths (list[int]): List of lengths for each chain

    Returns:
    tuple[float, float, float, list[float]]: A tuple containing:
        - pae_targets: mean PAE value for all target chains
        - pae_binder: mean PAE value for the binder chain
        - pae_interaction_mean: mean PAE value for all interactions
        - pae_interactions_per_target: list of mean PAE values for each target's interaction
    """
    binder_length = chain_lengths[0]
    
    # Calculate mean PAE for binder and all targets
    pae_complex = np.mean(pae)
    pae_targets = np.mean(pae[binder_length:, binder_length:])
    pae_binder = np.mean(pae[:binder_length, :binder_length])

    #! Old pae_interaction calculation
    # pae_interaction1 = np.mean(pae[:binder_length, binder_length:])
    # pae_interaction2 = np.mean(pae[binder_length:, :binder_length])
    # pae_interaction_old = (pae_interaction1 + pae_interaction2) / 2
    # print(f"pae_interaction_old: {pae_interaction_old}")

    # Calculate PAE for all targets and interactions
    start_pos = binder_length
    interaction_pae_values = []
    pae_interactions_per_target = []

    # Process each target chain separately
    for target_length in chain_lengths[1:]:  # Skip binder chain
        end_pos = start_pos + target_length
        
        # Flatten and concatenate interactions for this target
        interaction1 = pae[:binder_length, start_pos:end_pos].flatten()  # Binder to Target
        interaction2 = pae[start_pos:end_pos, :binder_length].flatten()  # Target to Binder
        interaction_pae_values.extend(np.concatenate([interaction1, interaction2]))

        # Mean PAE for this target's interactions
        pae_interactions_per_target.append(np.mean(interaction_pae_values) / 2)
        
        start_pos = end_pos

    # PAE for all interactions, weighted by number of targets
    pae_interaction = np.mean(interaction_pae_values) / (1 + len(chain_lengths[1:]))

    return pae_complex, pae_targets, pae_binder, pae_interaction, pae_interactions_per_target