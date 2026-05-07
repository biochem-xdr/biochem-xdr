import pickle
import random
import json
import networkx as nx
from datetime import datetime


def get_domain(node, G):
    return G.nodes[node].get("domain", "unknown")


def count_domain_crossings(path, G):
    if len(path) < 2:
        return 0
    domains = [get_domain(n, G) for n in path]
    return sum(
        1 for i in range(len(domains) - 1)
        if domains[i] != domains[i + 1]
        and domains[i] != "unknown"
        and domains[i + 1] != "unknown"
    )


def path_to_text(path, G):
    segments = []
    for i in range(len(path) - 1):
        node_a   = path[i]
        node_b   = path[i + 1]
        name_a   = G.nodes[node_a].get("display_name", str(node_a))
        name_b   = G.nodes[node_b].get("display_name", str(node_b))
        domain_a = get_domain(node_a, G)
        domain_b = get_domain(node_b, G)

        edge_data = G.get_edge_data(node_a, node_b)
        if edge_data:
            first_edge = list(edge_data.values())[0]
            edge_type  = first_edge.get("edge_type", "connected_to")
        else:
            edge_type = "connected_to"

        segments.append(
            f"{name_a} [{domain_a}] --{edge_type}--> {name_b} [{domain_b}]"
        )
    return "\n".join(segments)


def sample_paths(G, target_per_tier=2000, max_attempts=300000):
    """
    Sample cross-domain paths from the KG, stratified by domain
    crossing count.

    Tier boundaries from empirical graph analysis:
        T1: 1-2 crossings
        T2: 3 crossings
        T3: 4 crossings
        T4: 5+ crossings
    """
    usable_nodes = [
        node for node, data in G.nodes(data=True)
        if data.get("domain") not in ["unknown"]
        and data.get("display_name", "")
        and not str(data.get("display_name", "")).startswith("Reaction:")
        and not str(node).startswith("rhea:")
    ]

    domain_nodes = {}
    for node in usable_nodes:
        d = get_domain(node, G)
        domain_nodes.setdefault(d, []).append(node)

    print(f"Usable nodes: {len(usable_nodes)}")
    for d, nodes in domain_nodes.items():
        print(f"  {d:25}: {len(nodes)}")

    paths_by_tier = {"T1": [], "T2": [], "T3": [], "T4": []}
    seen_paths    = set()
    attempts      = 0
    last_print    = 0

    while attempts < max_attempts:
        attempts += 1

        if attempts - last_print >= 10000:
            counts = {t: len(v) for t, v in paths_by_tier.items()}
            print(f"  {attempts:,} attempts | "
                  f"T1:{counts['T1']} T2:{counts['T2']} "
                  f"T3:{counts['T3']} T4:{counts['T4']}")
            last_print = attempts

        if all(len(v) >= target_per_tier for v in paths_by_tier.values()):
            print("All tiers full")
            break

        needed = [t for t, v in paths_by_tier.items()
                  if len(v) < target_per_tier]
        target_tier = random.choice(needed)

        # Domain biasing — from empirical analysis, T4 paths are
        # more likely when starting at enzyme_kinetics and ending
        # at disease_mechanism or pathway_link
        if target_tier == "T1":
            src_d = random.choice(list(domain_nodes.keys()))
            tgt_d = random.choice(list(domain_nodes.keys()))
        elif target_tier == "T4":
            src_d = "enzyme_kinetics"
            tgt_d = random.choice(["disease_mechanism", "pathway_link"])
        else:
            src_d = random.choice(list(domain_nodes.keys()))
            tgt_d = random.choice(list(domain_nodes.keys()))

        if not domain_nodes.get(src_d) or not domain_nodes.get(tgt_d):
            continue

        start = random.choice(domain_nodes[src_d])
        end   = random.choice(domain_nodes[tgt_d])

        if start == end:
            continue

        try:
            path = nx.shortest_path(G, start, end)
        except nx.NetworkXNoPath:
            continue

        if len(path) < 3 or len(path) > 12:
            continue

        crossings = count_domain_crossings(path, G)
        if crossings <= 0:
            continue
        elif crossings <= 2:
            tier = "T1"
        elif crossings == 3:
            tier = "T2"
        elif crossings == 4:
            tier = "T3"
        else:
            tier = "T4"

        if len(paths_by_tier[tier]) >= target_per_tier:
            continue

        path_key = tuple(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        names = [G.nodes[n].get("display_name", "") for n in path]
        if not all(n.strip() for n in names):
            continue

        paths_by_tier[tier].append({
            "path":              path,
            "path_text":         path_to_text(path, G),
            "crossing_count":    crossings,
            "tier":              tier,
            "path_length":       len(path),
            "domains_traversed": [get_domain(n, G) for n in path],
            "start_domain":      get_domain(path[0], G),
            "end_domain":        get_domain(path[-1], G),
            "start_node":        path[0],
            "end_node":          path[-1],
        })

    return paths_by_tier


def save_paths(paths_by_tier, project_folder):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    paths_path = f"{project_folder}/biochem_xdr_paths_{timestamp}.pkl"
    stats_path = f"{project_folder}/biochem_xdr_paths_{timestamp}_stats.json"

    with open(paths_path, "wb") as f:
        pickle.dump(paths_by_tier, f)

    stats = {
        "timestamp":   timestamp,
        "total_paths": sum(len(v) for v in paths_by_tier.values()),
        "tiers":       {},
    }
    for tier, paths in paths_by_tier.items():
        if not paths:
            continue
        stats["tiers"][tier] = {
            "count":         len(paths),
            "avg_length":    round(
                sum(p["path_length"] for p in paths) / len(paths), 2
            ),
            "avg_crossings": round(
                sum(p["crossing_count"] for p in paths) / len(paths), 2
            ),
            "min_crossings": min(p["crossing_count"] for p in paths),
            "max_crossings": max(p["crossing_count"] for p in paths),
            "domain_pairs":  {},
        }
        for p in paths:
            pair = f"{p['start_domain']} -> {p['end_domain']}"
            stats["tiers"][tier]["domain_pairs"][pair] = (
                stats["tiers"][tier]["domain_pairs"].get(pair, 0) + 1
            )

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved: {paths_path}")
    print(f"Saved: {stats_path}")
    return paths_path, stats_path


if __name__ == "__main__":
    import sys

    graph_path     = sys.argv[1]
    project_folder = sys.argv[2] if len(sys.argv) > 2 else "."

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    paths_by_tier = sample_paths(G, target_per_tier=2000)

    total = sum(len(v) for v in paths_by_tier.values())
    print(f"\nTotal paths: {total}")
    for tier, paths in paths_by_tier.items():
        if paths:
            avg_l = sum(p["path_length"] for p in paths) / len(paths)
            avg_c = sum(p["crossing_count"] for p in paths) / len(paths)
            print(f"  {tier}: {len(paths)} paths | "
                  f"avg length {avg_l:.1f} | avg crossings {avg_c:.1f}")

    save_paths(paths_by_tier, project_folder)