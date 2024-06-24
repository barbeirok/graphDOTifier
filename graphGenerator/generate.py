import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import words
import subprocess
import os

# Download the words corpus if not already downloaded
nltk.download('words')

# Get a list of English words
english_words = words.words()


def generate_random_dag(nodes, edges):
    """
    Generates a random Directed Acyclic Graph (DAG) with the given number of nodes and edges.

    Parameters:
        nodes (int): Number of nodes in the graph.
        edges (int): Number of edges in the graph.

    Returns:
        nx.DiGraph: Random DAG.
    """
    if edges > nodes * (nodes - 1) // 2:
        return None

    # Generate a random DAG
    dag = nx.DiGraph()

    # Add nodes to the graph with random English words as labels
    for i in range(nodes):
        dag.add_node(i, label=random.choice(english_words))

    # Add random edges until desired number of edges is reached
    while dag.number_of_edges() < edges:
        # Generate random edge
        edge = (random.randint(0, nodes - 1), random.randint(0, nodes - 1))
        # Ensure the edge is valid and not creating a cycle
        if edge[0] != edge[1] and not nx.has_path(dag, edge[1], edge[0]):
            weight = random.randint(1, 10)  # Assign a random weight to the edge
            dag.add_edge(*edge, weight=weight)

    return dag


# Create the DFG directory if it doesn't exist
os.makedirs("horizontalDFG", exist_ok=True)

for i in range(250):
    num_nodes = random.randint(2, 5)
    max_edges = math.floor(num_nodes * (num_nodes - 1) // 2)
    num_edges = random.randint(1, max_edges)
    print("\n_____________________________")
    print(f"DFG: {i}")
    print(f"Nº of nodes: {num_nodes}")
    print(f"Nº of edges: {num_edges}")
    random_dag = generate_random_dag(num_nodes, num_edges)
    if random_dag is None:
        continue
    # Save the random DAG as a Graphviz DOT file with edge labels
    dot_filename = f"horizontalDFG/dfg{i}.dot"
    with open(dot_filename, 'w') as dot_file:
        dot_file.write("digraph G {\n")
        dot_file.write("    rankdir=LR;\n")
        for u, v, data in random_dag.edges(data=True):
            dot_file.write(f"    {u} [label=\"{random_dag.nodes[u]['label']}\"];\n")
            dot_file.write(f"    {v} [label=\"{random_dag.nodes[v]['label']}\"];\n")
            dot_file.write(f"    {u} -> {v} [label={data['weight']}];\n")
        dot_file.write("}")

    # Convert the DOT file to PNG using Graphviz
    png_filename = f"horizontalDFG/dfg{i}.png"
    result = subprocess.run(["dot", "-Tpng", dot_filename, "-o", png_filename], capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print(f"Error generating PNG for {dot_filename}: {result.stderr}")
    else:
        print(f"Successfully generated {png_filename}")
