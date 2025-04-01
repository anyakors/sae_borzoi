import numpy as np
import pandas as pd
import gget
import argparse
import os
from tqdm import tqdm
import json

# add argparse and read the input tomtom folder

parser = argparse.ArgumentParser(description='Analyze tomtom results')
parser.add_argument('--input', type=str, help='Path to the meme/tomtom folder')

def main():
    args = parser.parse_args()

    input_folder = args.input 

    cisbp_map = json.load(open('/Users/anya/gworkspace/sae_torch/models/cisbp_motif_gene_map.json', 'r'))

    for ind,node_id in tqdm(enumerate(os.listdir(os.path.join(input_folder, 'tomtom_out')))):
        node_tomtom = os.path.join(input_folder, 'tomtom_out', node_id, 'tomtom.tsv')
        node_meme = os.path.join(input_folder, 'meme_out', node_id, 'motifs.tsv')
        if os.path.exists(node_tomtom) and os.path.exists(node_meme):
            df_t = pd.read_csv(node_tomtom, sep='\t', index_col=None)
            df_m = pd.read_csv(node_meme, sep='\t', index_col=None)
            df_m = df_m[df_m['e_value'] < 0.05]
            df_t = df_t[df_t['q-value'] < 0.05]
            df_t = df_t[df_t['E-value'] < 0.05]
            # only keep the motifs that are in the meme file
            df_t = df_t[df_t['Query_ID'].isin(df_m['sequence'])]
            if len(df_t) == 0:
                continue
            if '2.00' in df_t['Target_ID'].iloc[0]:
                df_t['Gene'] = [cisbp_map[x] if x in cisbp_map.keys() else None for x in df_t['Target_ID']]
            else:
                df_t['Gene'] = [x.split('.')[0] for x in df_t['Target_ID']]
            df_t['Node'] = [node_id]*len(df_t)

            genes = list(df_t['Gene'].unique())
            database = "ontology"

            try:
                df_enr = gget.enrichr(genes, database=database) #, ensembl=True
            except AttributeError:
                print('AttributeError with node_id: ', node_id)
                print(df_t.head())
            # filter by pval
            df_enr = df_enr[df_enr['p_val'] < 0.05]
            df_enr.to_csv(os.path.join(input_folder, 'tomtom_out', node_id, 'enrichr_ontology.tsv'))

            # for each row in df_t, check whether it's in the 'overlapping_genes' column of df_enr
            # if it is, then add the corresponding 'term_name' to the df_t
            # if it's not, then add 'None'
            terms = []
            for j in range(len(df_t)):
                gene = df_t.iloc[j]['Gene']
                terms_j = None
                for it in range(len(df_enr)):
                    if gene in df_enr.iloc[it]['overlapping_genes']:
                        terms_j = df_enr.iloc[it]['path_name']
                        break
                if terms_j is None:
                    terms.append(None)
                else:
                    terms.append(terms_j)

            df_t['Term'] = terms

            df_t.to_csv(os.path.join(input_folder, 'tomtom_out', node_id, 'tomtom_filtered.tsv'), sep='\t', index=False)

            if ind==0:
                df_all = df_t[['Query_ID', 'Gene', 'Node', 'q-value', 'E-value', 'Query_consensus', 'Term']]
            else:
                df_all = pd.concat([df_all, df_t])
    
    df_all.to_csv(os.path.join(input_folder, 'tomtom_out', 'tomtom_all.tsv'), sep='\t', index=False)


if __name__ == "__main__":
    main()

