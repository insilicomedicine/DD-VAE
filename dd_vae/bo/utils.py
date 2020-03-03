import networkx as nx
from rdkit.Chem import rdmolops


def max_ring_penalty(mol):
    cycle_list = nx.cycle_basis(
        nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    return -max(0, cycle_length - 6)
