# KG construction parameters
# Organism and taxonomy filters are hard constraints —
# changing these changes the dataset, not just performance.

ORGANISM_CODE = "hsa"       # KEGG organism code — Homo sapiens only
TAXONOMY_ID   = 9606        # UniProt taxonomy — Homo sapiens

KEGG_REQUEST_DELAY   = 0.3  # seconds between KEGG API calls
UNIPROT_REQUEST_DELAY = 0.5  # seconds between UniProt API calls

# UniProt query: only reviewed (Swiss-Prot) entries
UNIPROT_REVIEWED_ONLY = True

# Maximum proteins fetched per EC number from UniProt
UNIPROT_MAX_RESULTS = 10

# Tissue specificity text truncation
TISSUE_TEXT_MAX_LEN = 500

# Progress reporting interval
PROGRESS_INTERVAL = 50