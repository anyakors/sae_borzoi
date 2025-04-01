import re
import json

def parse_meme_file(file_path):
    """
    Parse a MEME file and extract motif IDs and gene names.
    
    Args:
        file_path (str): Path to the MEME format file
        
    Returns:
        dict: Dictionary with motif IDs as keys and gene names as values
    """
    motif_dict = {}
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Regular expression to match MOTIF lines
        # Format: MOTIF M01204_2.00 TIGD1
        motif_pattern = r'MOTIF (M\d+_\d+\.\d+) (\w+)'

        # Find all matches in the file
        matches = re.finditer(motif_pattern, content)
        
        for match in matches:
            motif_id = match.group(1)  # M00111_2.00
            gene_name = match.group(2).split(')_')[0]  # TFAP2D
            motif_dict[motif_id] = gene_name
    
    return motif_dict

def save_to_json(motif_dict, output_file):
    """
    Save the motif dictionary to a JSON file.
    
    Args:
        motif_dict (dict): Dictionary with motif IDs and gene names
        output_file (str): Path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(motif_dict, f, indent=2)

# Example usage
if __name__ == "__main__":
    input_file = "/Users/anya/Downloads/Homo_sapiens.meme"
    output_file = "/Users/anya/gworkspace/sae_torch/models/cisbp_motif_gene_map.json"
    
    # Parse MEME file and create dictionary
    motif_gene_map = parse_meme_file(input_file)
    
    # Save to JSON file
    save_to_json(motif_gene_map, output_file)
    
    # Print the results
    print("Motif ID to Gene Name mapping:")
    for motif_id, gene_name in motif_gene_map.items():
        print(f"{motif_id}: {gene_name}")
