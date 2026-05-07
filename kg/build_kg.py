import time
import pickle
import json
import requests
import xml.etree.ElementTree as ET
import networkx as nx
from bioservices import KEGG
from datetime import datetime


def fetch_all_hsa_pathways(k):
    pathway_list = k.list("pathway", "hsa")
    pathway_ids = []
    for line in str(pathway_list).strip().split("\n"):
        if line:
            pid = line.split("\t")[0].replace("path:", "")
            pathway_ids.append(pid)
    return pathway_ids


def parse_kgml_into_graph(G, pathway_id, kgml_str, pathway_title):
    root = ET.fromstring(kgml_str)

    entries = {}
    for entry in root.findall("entry"):
        eid     = entry.attrib.get("id")
        enames  = entry.attrib.get("name", "")
        etype   = entry.attrib.get("type")
        graphics = entry.find("graphics")
        display  = graphics.attrib.get("name", "") if graphics is not None else ""

        entries[eid] = {
            "names":   enames,
            "type":    etype,
            "display": display,
        }

        node_id = display if display else enames
        if not G.has_node(node_id):
            domain = "enzyme_kinetics" if etype == "gene" else "metabolic_pathway"
            G.add_node(
                node_id,
                entry_type=etype,
                domain=domain,
                source_db="KEGG",
                kegg_names=enames,
                pathway=pathway_title,
            )

    for relation in root.findall("relation"):
        entry1 = relation.attrib.get("entry1")
        entry2 = relation.attrib.get("entry2")
        for subtype in relation.findall("subtype"):
            rel_name = subtype.attrib.get("name")
            n1_info  = entries.get(entry1, {})
            n2_info  = entries.get(entry2, {})
            n1 = n1_info.get("display") or n1_info.get("names", "")
            n2 = n2_info.get("display") or n2_info.get("names", "")
            if n1 and n2 and G.has_node(n1) and G.has_node(n2):
                G.add_edge(n1, n2, relation=rel_name, source="KEGG",
                           pathway=pathway_title)

    for reaction in root.findall("reaction"):
        rname     = reaction.attrib.get("name", "")
        rtype     = reaction.attrib.get("type", "")
        substrates = [s.attrib.get("name") for s in reaction.findall("substrate")]
        products   = [p.attrib.get("name") for p in reaction.findall("product")]

        # Find the gene entry linked to this reaction
        gene_node = None
        for entry in root.findall("entry"):
            if entry.attrib.get("reaction") == reaction.attrib.get("name"):
                graphics = entry.find("graphics")
                gene_node = (
                    graphics.attrib.get("name")
                    if graphics is not None
                    else entry.attrib.get("name", "")
                )
                break

        for s in substrates:
            if s and not G.has_node(s):
                G.add_node(s, entry_type="compound", domain="metabolic_pathway",
                           source_db="KEGG")
        for p in products:
            if p and not G.has_node(p):
                G.add_node(p, entry_type="compound", domain="metabolic_pathway",
                           source_db="KEGG")

        if gene_node and G.has_node(gene_node):
            for s in substrates:
                if s and G.has_node(s):
                    G.add_edge(s, gene_node, relation="substrate_of",
                               source="KEGG", reaction=rname)
            for p in products:
                if p and G.has_node(p):
                    G.add_edge(gene_node, p, relation="produces",
                               source="KEGG", reaction=rname)


def build_kegg_graph(k):
    G = nx.MultiDiGraph()
    pathway_ids = fetch_all_hsa_pathways(k)
    print(f"Fetched {len(pathway_ids)} hsa pathways")

    failed = []
    for i, pid in enumerate(pathway_ids):
        try:
            kgml = k.get(pid, "kgml")
            if not kgml or kgml == 404:
                failed.append(pid)
                continue
            pathway_info = k.get(pid)
            title = pid
            for line in str(pathway_info).split("\n"):
                if line.startswith("NAME"):
                    title = line.split(None, 1)[1].strip()
                    break

            # Add pathway node
            if not G.has_node(title):
                G.add_node(title, entry_type="map", domain="pathway_link",
                           source_db="KEGG", pathway_id=pid)

            parse_kgml_into_graph(G, pid, kgml, title)
            time.sleep(0.3)

        except Exception:
            failed.append(pid)
            continue

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pathway_ids)} pathways | "
                  f"{G.number_of_nodes()} nodes | "
                  f"{G.number_of_edges()} edges")

    print(f"Failed: {len(failed)}")
    return G


def integrate_rhea(G):
    # Fetch all Rhea reactions via REST
    url = "https://www.rhea-db.org/rhea?query=&columns=rhea-id,equation,ec&format=tsv&limit=100000"
    response = requests.get(url, timeout=60)
    lines = response.text.strip().split("\n")[1:]  # skip header
    print(f"Fetched {len(lines)} Rhea reactions")

    added_reactions = 0
    added_edges     = 0

    for line in lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rhea_id  = parts[0].strip()
        equation = parts[1].strip() if len(parts) > 1 else ""
        ec_list  = parts[2].strip() if len(parts) > 2 else ""

        if not equation:
            continue

        # Parse left and right sides of equation
        sides = equation.split(" = ")
        if len(sides) != 2:
            continue
        left_cpds  = [c.strip() for c in sides[0].split(" + ")]
        right_cpds = [c.strip() for c in sides[1].split(" + ")]

        reaction_node = f"rhea:{rhea_id}"
        if not G.has_node(reaction_node):
            G.add_node(reaction_node, entry_type="reaction",
                       domain="metabolic_pathway", source_db="Rhea",
                       ec=ec_list, equation=equation)
            added_reactions += 1

        for cpd in left_cpds + right_cpds:
            if cpd and not G.has_node(cpd):
                G.add_node(cpd, entry_type="compound",
                           domain="metabolic_pathway", source_db="Rhea")

        for cpd in left_cpds:
            if cpd:
                G.add_edge(cpd, reaction_node, relation="substrate",
                           source="Rhea")
                added_edges += 1
        for cpd in right_cpds:
            if cpd:
                G.add_edge(reaction_node, cpd, relation="product",
                           source="Rhea")
                added_edges += 1

        # Link to KEGG enzyme nodes by EC number
        if ec_list:
            for node, data in G.nodes(data=True):
                if data.get("source_db") == "KEGG" and \
                        data.get("domain") == "enzyme_kinetics":
                    node_names = data.get("kegg_names", "")
                    if ec_list in node_names:
                        G.add_edge(node, reaction_node,
                                   relation="has_reaction", source="Rhea")
                        added_edges += 1

    print(f"Rhea: {added_reactions} reaction nodes, {added_edges} edges")
    return G


def integrate_uniprot(G, k):
    # Collect EC numbers from KEGG enzyme nodes
    ec_numbers = set()
    for node, data in G.nodes(data=True):
        if data.get("domain") == "enzyme_kinetics":
            names = data.get("kegg_names", "")
            for part in names.split():
                if part.startswith("ec:"):
                    ec_numbers.add(part.replace("ec:", ""))

    ec_numbers = list(ec_numbers)
    print(f"Unique EC numbers to query: {len(ec_numbers)}")
    print("Querying UniProt for disease associations...")
    print("(This takes 20-30 minutes for full coverage)")

    stats = {
        "ec_numbers_queried": 0,
        "proteins_found":     0,
        "disease_nodes_added": 0,
        "disease_edges_added": 0,
        "tissue_annotations":  0,
        "uniprot_ids_added":   0,
    }

    for i, ec in enumerate(ec_numbers):
        url = (
            f"https://rest.uniprot.org/uniprotkb/search?"
            f"query=ec:{ec}+AND+organism_id:9606+AND+reviewed:true"
            f"&fields=accession,gene_names,protein_name,organism_name,"
            f"ec,cc_disease,cc_tissue_specificity"
            f"&format=json&size=10"
        )
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                continue
            data = response.json()
            results = data.get("results", [])
            stats["proteins_found"] += len(results)
            stats["ec_numbers_queried"] += 1

            for protein in results:
                accession = protein.get("primaryAccession", "")
                gene_info = protein.get("genes", [])
                gene_name = ""
                if gene_info:
                    gene_name = gene_info[0].get("geneName", {}).get("value", "")

                stats["uniprot_ids_added"] += 1

                # Disease associations
                comments = protein.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") != "DISEASE":
                        continue
                    disease_info = comment.get("disease", {})
                    disease_name = disease_info.get("diseaseId", "")
                    if not disease_name:
                        continue

                    if not G.has_node(disease_name):
                        G.add_node(
                            disease_name,
                            entry_type="disease",
                            domain="disease_mechanism",
                            source_db="UniProt",
                            uniprot_accession=accession,
                        )
                        stats["disease_nodes_added"] += 1

                    # Connect enzyme node to disease node
                    matching_nodes = [
                        n for n, d in G.nodes(data=True)
                        if d.get("domain") == "enzyme_kinetics"
                        and (ec in d.get("kegg_names", "")
                             or gene_name in n)
                    ]
                    for node in matching_nodes:
                        G.add_edge(
                            node, disease_name,
                            relation="associated_with_disease",
                            source="UniProt",
                            ec=ec,
                        )
                        G.add_edge(
                            disease_name, node,
                            relation="has_enzyme_association",
                            source="UniProt",
                            ec=ec,
                        )
                        stats["disease_edges_added"] += 2

                    # Tissue specificity annotation
                    for comment in comments:
                        if comment.get("commentType") != "TISSUE SPECIFICITY":
                            continue
                        texts = comment.get("texts", [])
                        tissue_text = "; ".join(
                            t.get("value", "") for t in texts
                        )
                        for node in matching_nodes:
                            G.nodes[node]["tissue_specificity"] = \
                                tissue_text[:500]
                            stats["tissue_annotations"] += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(ec_numbers)} EC numbers | "
                      f"Diseases: {stats['disease_nodes_added']} | "
                      f"Edges: {stats['disease_edges_added']}")

            time.sleep(0.5)

        except Exception:
            continue

    return stats


def save_graph(G, project_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pickle_path = (
        f"{project_folder}/"
        f"biochem_xdr_kg_kegg_rhea_uniprot_{timestamp}.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)

    stats = {
        "timestamp":   timestamp,
        "sources":     ["KEGG", "Rhea", "UniProt"],
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types":  {},
        "domains":     {},
        "source_dbs":  {},
    }
    for node, data in G.nodes(data=True):
        for key, field in [
            ("node_types", "entry_type"),
            ("domains",    "domain"),
            ("source_dbs", "source_db"),
        ]:
            val = data.get(field, "unknown")
            stats[key][val] = stats[key].get(val, 0) + 1

    stats_path = pickle_path.replace(".pkl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved: {pickle_path}")
    print(f"Saved: {stats_path}")
    return pickle_path, stats_path


if __name__ == "__main__":
    import sys
    project_folder = sys.argv[1] if len(sys.argv) > 1 else "."

    k = KEGG()
    print("Building KEGG graph...")
    G = build_kegg_graph(k)
    print(f"After KEGG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nIntegrating Rhea...")
    G = integrate_rhea(G)
    print(f"After Rhea: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nIntegrating UniProt...")
    stats = integrate_uniprot(G, k)
    print(f"After UniProt: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for k_stat, v in stats.items():
        print(f"  {k_stat}: {v}")

    save_graph(G, project_folder)