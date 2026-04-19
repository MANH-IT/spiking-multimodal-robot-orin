import networkx as nx

def maximum_spanning_arborescence(weights):
    # weights: 2D numpy array, weights[i][j] = score(i->j)
    n = weights.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, weight=weights[i,j])
    mst = nx.maximum_spanning_arborescence(G, attr='weight')
    return mst
