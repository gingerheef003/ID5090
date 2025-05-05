import pickle
from collections import Counter

def find_most_common_gene_pattern(last_gen_path):
    """Find and display the most common gene pattern in the last generation gene pool."""
    # Load the last generation gene pool
    with open(last_gen_path, 'rb') as f:
        last_gen_pool = pickle.load(f)
    
    # Flatten the gene pool into a list of genes
    all_genes = [gene for creature_genes in last_gen_pool for gene in creature_genes]
    
    # Count the occurrences of each gene
    gene_counter = Counter(all_genes)
    
    # Find the most common gene pattern
    most_common_genes = gene_counter.most_common(10)  # Top 5 most common genes
    
    # Display the results
    print("Most Common Gene Patterns:")
    for gene, count in most_common_genes:
        gene_str = "-".join(map(str, gene))  # Convert gene tuple to a string
        print(f"{gene_str}: {count} occurrences")

def group_genes_by_type(last_gen_path):
    """Group genes by type (ignoring weight) and quantify core circuits."""
    # Load the last generation gene pool
    with open(last_gen_path, 'rb') as f:
        last_gen_pool = pickle.load(f)
    
    # Flatten the gene pool into a list of genes
    all_genes = [gene for creature_genes in last_gen_pool for gene in creature_genes]
    
    # Group genes by type (ignoring weight)
    gene_types = [gene[:3] for gene in all_genes]  # Extract the first three elements of each gene
    gene_type_counter = Counter(gene_types)
    
    # Display the results
    print("Gene Type Frequencies:")
    for gene_type, count in gene_type_counter.most_common(10):  # Top 10 most common gene types
        gene_type_str = "-".join(map(str, gene_type))  # Convert gene type tuple to a string
        print(f"{gene_type_str}: {count} occurrences")

# Example usage
if __name__ == "__main__":
    last_gen_path = 'Project/Data/last_generation_gene_pool 20250503_01_55_04.pkl'  # Replace with actual path
    find_most_common_gene_pattern(last_gen_path)
    group_genes_by_type(last_gen_path)