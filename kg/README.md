# KG Construction

Builds the integrated human biochemical knowledge graph from KEGG, Rhea, and UniProt.

## Requirements

```bash
pip install bioservices networkx pandas requests lxml
```

## Usage

```bash
python kg/build_kg.py /path/to/output/folder
```

This runs all three stages sequentially and saves a pickle and stats JSON to the output folder. Expect 4-6 hours wall time — the UniProt stage queries ~6,000 EC numbers with rate limiting.

## What it builds

**KEGG** — all hsa (human) pathways fetched in KGML format. Gene entries become `enzyme_kinetics` nodes, compounds become `metabolic_pathway` nodes, map entries become `pathway_link` nodes. ECrel and other relation subtypes are preserved as edge attributes.

**Rhea** — all reactions fetched via the Rhea REST API. Each reaction becomes a node with substrate and product edges. Reactions are linked to KEGG enzyme nodes by EC number where matches exist.

**UniProt** — disease associations and tissue specificity fetched for all EC numbers found in the KEGG graph. Only reviewed Swiss-Prot entries (taxonomy 9606) are used. Disease nodes become `disease_mechanism` nodes connected to enzyme nodes by `associated_with_disease` / `has_enzyme_association` edges.

## Output