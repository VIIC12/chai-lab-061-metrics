import freesasa
import numpy as np
from Bio import PDB

# Reference values from Wilke et al. (2013) - Empirical
MAX_ASA_REF = {
    'ALA': 121.0,
    'ARG': 265.0,
    'ASN': 187.0,
    'ASP': 187.0,
    'CYS': 148.0,
    'GLU': 214.0,
    'GLN': 214.0,
    'GLY': 97.0,
    'HIS': 216.0,
    'ILE': 195.0,
    'LEU': 191.0,
    'LYS': 230.0,
    'MET': 203.0,
    'PHE': 228.0,
    'PRO': 154.0,
    'SER': 143.0,
    'THR': 163.0,
    'TRP': 264.0,
    'TYR': 255.0,
    'VAL': 165.0,
}


def calculate_surface_metrics(cif_out_path, binder_chain="A", threshold=0.2) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    
    # Convert CIF to PDB
    parser = PDB.MMCIFParser()
    io = PDB.PDBIO()
    base_path = str(cif_out_path).rsplit('.cif', 1)[0]
    print(f"cif_out_path: {base_path}")

    cif_to_pdb_structure = parser.get_structure('structure', cif_out_path)
    io.set_structure(cif_to_pdb_structure)
    structure = f'{base_path}.pdb'
    io.save(structure)

    # Set up the classifier and parameters
    classifier = freesasa.Classifier.getStandardClassifier('naccess')
    parameters = freesasa.Parameters({'algorithm': freesasa.LeeRichards, 'n-slices': 100})

    # Calculate the SASA, return result object
    structure_obj = freesasa.Structure(structure, classifier)
    sasa_calc_results = freesasa.calc(structure_obj, parameters)

    # Get total SASA
    total_sasa = sasa_calc_results.totalArea()

    # Initialize dictionaries to store results for each chain
    sasa_per_residue = {}
    rasa_arrays = {}

    # Process each chain
    for chain in sasa_calc_results.residueAreas().keys():
        # Get SASA of current chain
        sasa_subselection = freesasa.selectArea([f'chain_{chain}, Chain {chain}'], structure_obj, sasa_calc_results)
        
        # Get number of residues in chain
        num_residues = len(sasa_calc_results.residueAreas()[chain].keys())
        
        # Calculate SASA per residue for this chain, average of all residues in the chain
        sasa_per_residue[chain] = sasa_subselection[f'chain_{chain}'] / num_residues

        # Calculate RASA for each residue in chain
        rasa_values = []
        residue_numbers = list(sasa_calc_results.residueAreas()[chain].keys())
        for residue_num in residue_numbers:
            asa = sasa_calc_results.residueAreas()[chain][residue_num]
            max_asa = MAX_ASA_REF[asa.residueType]
            rasa = asa.total / max_asa if max_asa > 0 else 0
            rasa_values.append(rasa)

        rasa_arrays[chain] = np.array(rasa_values)
        #print(f"Chain {chain} RASA values: {rasa_arrays[chain]}")
        
    # threshold of 0.2 (or 20% relative accessibility) is an empirically determined value.
    # based on studies that have found this level oxf exposure to be a good indicator of potential disorder.


    # Calculate overall RASA for the complex
    all_rasa_values = np.concatenate(list(rasa_arrays.values()))
    fraction_disordered_complex = np.mean(all_rasa_values > threshold)
    #print(f"All RASA values: {all_rasa_values}")  # Add this debug line
    #print(f"Fraction disordered (complex): {fraction_disordered_complex}")

    # Calculate fraction disordered for binder chain only
    binder_rasa_values = rasa_arrays[binder_chain]
    #print(f"Binder chain RASA values: {binder_rasa_values}")  # Add this debug line
    fraction_disordered_binder = np.mean(binder_rasa_values > threshold)
    #print(f"Fraction disordered (binder chain {binder_chain}): {fraction_disordered_binder}")

    return sasa_per_residue, rasa_arrays, fraction_disordered_binder, fraction_disordered_complex