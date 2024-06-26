# #Python version: 3.11.5
# #Numpy version: 1.26.4
# #Scipy version: 1.13.1
# #Scanpy version: 1.10.1

# #run function:
# #after sc.pp.neighbors
# import paga_weight as pw
# pw.paga(adata,root,connectivities_threshold)

# #draw the plot:
# sc.pl.paga(adata, color = "connectivities_tree")
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree



from scanpy import _utils #find and combine
from scanpy.neighbors import Neighbors #find and combine

def paga(
    adata: AnnData,
    groups: str | None = None,
    *,
    neighbors_key: str | None = None,
    copy: bool = False,
    root: str | None = None, #add
    connectivities_threshold: float | None = None, #add
) -> AnnData | None:
    """\
    Mapping out the coarse-grained connectivity structures of complex manifolds :cite:p:`Wolf2019`.

    By quantifying the connectivity of partitions (groups, clusters) of the
    single-cell graph, partition-based graph abstraction (PAGA) generates a much
    simpler abstracted graph (*PAGA graph*) of partitions, in which edge weights
    represent confidence in the presence of connections. By thresholding this
    confidence in :func:`~scanpy.pl.paga`, a much simpler representation of the
    manifold data is obtained, which is nonetheless faithful to the topology of
    the manifold.

    The confidence should be interpreted as the ratio of the actual versus the
    expected value of connections under the null model of randomly connecting
    partitions. We do not provide a p-value as this null model does not
    precisely capture what one would consider "connected" in real data, hence it
    strongly overestimates the expected value. See an extensive discussion of
    this in :cite:t:`Wolf2019`.

    .. note::
        Note that you can use the result of :func:`~scanpy.pl.paga` in
        :func:`~scanpy.tl.umap` and :func:`~scanpy.tl.draw_graph` via
        `init_pos='paga'` to get single-cell embeddings that are typically more
        faithful to the global topology.

    Parameters
    ----------
    adata
        An annotated data matrix.
    groups
        Key for categorical in `adata.obs`. You can pass your predefined groups
        by choosing any categorical annotation of observations. Default:
        The first present key of `'leiden'` or `'louvain'`.
    neighbors_key
        If not specified, paga looks `.uns['neighbors']` for neighbors settings
        and `.obsp['connectivities']`, `.obsp['distances']` for connectivities and
        distances respectively (default storage places for `pp.neighbors`).
        If specified, paga looks `.uns[neighbors_key]` for neighbors settings and
        `.obsp[.uns[neighbors_key]['connectivities_key']]`,
        `.obsp[.uns[neighbors_key]['distances_key']]` for connectivities and distances
        respectively.
    copy
        Copy `adata` before computation and return a copy. Otherwise, perform
        computation inplace and return `None`.
    root
        Use a root to refine the paga. This root must be consistent with
        the root used to draw the dpt. If not specified, this operation is not
        performed by default
    connectivities_threshold
        default is 0.5.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.uns['connectivities']` : :class:`numpy.ndarray` (dtype `float`)
        The full adjacency matrix of the abstracted graph, weights correspond to
        confidence in the connectivities of partitions.
    `adata.uns['connectivities_tree']` : :class:`scipy.sparse.csr_matrix` (dtype `float`)
        The adjacency matrix of the tree-like subgraph that best explains
        the topology.

    Notes
    -----
    Together with a random walk-based distance measure
    (e.g. :func:`scanpy.tl.dpt`) this generates a partial coordinatization of
    data useful for exploring and explaining its variation.

    .. currentmodule:: scanpy

    See Also
    --------
    pl.paga
    pl.paga_path
    pl.paga_compare
    """
    check_neighbors = "neighbors" if neighbors_key is None else neighbors_key
    if check_neighbors not in adata.uns:
        raise ValueError(
            "You need to run `pp.neighbors` first to compute a neighborhood graph."
        )
    if groups is None:
        for k in ("leiden", "louvain"):
            if k in adata.obs.columns:
                groups = k
                break
    if groups is None:
        raise ValueError(
            "You need to run `tl.leiden` or `tl.louvain` to compute "
            "community labels, or specify `groups='an_existing_key'`"
        )
    elif groups not in adata.obs.columns:
        raise KeyError(f"`groups` key {groups!r} not found in `adata.obs`.")

    # 检查 root 是否在指定的 groups 列中
    if root is not None and root not in adata.obs[groups].values:
        raise ValueError(f"The root value '{root}' is not found in the specified group '{groups}'.")
    if root is not None and connectivities_threshold is None:
        connectivities_threshold = 0.5
        print("connectivities_threshold has not input, default value is 0.5")

    adata = adata.copy() if copy else adata
    _utils.sanitize_anndata(adata)
    paga = PAGA(adata, groups=groups, root=root,connectivities_threshold=connectivities_threshold, neighbors_key=neighbors_key)  # 传递 root 参数并使用关键字参数形式传递 neighbors_key
    # only add if not present
    if "paga" not in adata.uns:
        adata.uns["paga"] = {}
    paga.compute_connectivities()
    adata.uns["paga"]["connectivities"] = paga.connectivities
    adata.uns["paga"]["connectivities_tree"] = paga.connectivities_tree
    # adata.uns['paga']['expected_n_edges_random'] = paga.expected_n_edges_random
    adata.uns[groups + "_sizes"] = np.array(paga.ns)
    adata.uns["paga"]["groups"] = groups
    return adata if copy else None


class PAGA:
    def __init__(self, adata, groups, root, connectivities_threshold, neighbors_key=None):
        assert groups in adata.obs.columns
        self._adata = adata
        self._neighbors = Neighbors(adata, neighbors_key=neighbors_key)
        self._groups_key = groups
        self._root = root  # add
        self._threshold = connectivities_threshold

    def compute_connectivities(self):
        return self._compute_connectivities_v1_2()

    def _compute_connectivities_v1_2(self):
        import igraph

        distance = self._neighbors.distances.copy()
        def normalize_sparse_matrix(matrix):
            #normalizing sparse matrix
            from scipy.sparse import csr_matrix
            non_zero_values = matrix.data
            current_mean = np.mean(non_zero_values)
            scale_factor = 1 / current_mean
            normalized_values = non_zero_values * scale_factor
            normalized_matrix = csr_matrix((normalized_values, matrix.indices, matrix.indptr), shape=matrix.shape)
            return normalized_matrix
        distance = normalize_sparse_matrix(distance) #Normalize sparse matrix
        # should be directed if we deal with distances
        g = _utils.get_igraph_from_adjacency(distance, directed=True)
        vc = igraph.VertexClustering(
            g, membership=self._adata.obs[self._groups_key].cat.codes.values
        )
        ns = vc.sizes()
        n = sum(ns)
        es_inner_cluster = [vc.subgraph(i).ecount() for i in range(len(ns))]
        cg = vc.cluster_graph(combine_edges="sum")
        inter_es = cg.get_adjacency_sparse(attribute="weight")
        es = np.array(es_inner_cluster) + inter_es.sum(axis=1).A1
        inter_es = inter_es + inter_es.T  # \epsilon_i + \epsilon_j
        connectivities = inter_es.copy()
        expected_n_edges = inter_es.copy()
        inter_es = inter_es.tocoo()
        for i, j, v in zip(inter_es.row, inter_es.col, inter_es.data):
            expected_random_null = (es[i] * ns[j] + es[j] * ns[i]) / (n - 1)
            if expected_random_null != 0:
                scaled_value = v / expected_random_null
            else:
                scaled_value = 1
            if scaled_value > 1:
                scaled_value = 1
            connectivities[i, j] = scaled_value
            expected_n_edges[i, j] = expected_random_null
        # set attributes
        self.ns = ns
        self.expected_n_edges_random = expected_n_edges
        self.connectivities = connectivities
        self.connectivities_tree = self._get_connectivities_tree_v1_2()
        return inter_es.tocsr(), connectivities

    def _get_connectivities_tree_v1_2(self):
        inverse_connectivities = self.connectivities.copy()
        inverse_connectivities.data = 1.0 / inverse_connectivities.data
        connectivities_tree = minimum_spanning_tree(inverse_connectivities)

        # Find root index save as root_index
        labels = self._adata.obs[self._groups_key]
        unique_labels = sorted(labels.unique())
        if self._root in unique_labels:
            root_index = unique_labels.index(self._root)
        else:
            raise ValueError(f"{self._root} not in the {self._groups_key}")

        # Find the different groups of levels
        from collections import deque, defaultdict
        def build_graph_and_traverse(connectivities_tree, root_index = root_index):

            graph = defaultdict(list)
            connectivities_tree_tocoo = connectivities_tree.tocoo()
            edges = list(zip(connectivities_tree_tocoo.row, connectivities_tree_tocoo.col))
            

            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            

            queue = deque([root_index])
            levels = {root_index: 0}
            predecessors = {root_index: -99}
            
            while queue:
                current = queue.popleft()
                current_level = levels[current]
                for neighbor in graph[current]:
                    if neighbor not in levels:
                        queue.append(neighbor)
                        levels[neighbor] = current_level + 1
                        predecessors[neighbor] = current
            
            return levels, predecessors
        
        # Calculate the connectivity of all points to their predecessor points and
        # the connectivity of the predecessor points of this point and its predecessor
        # points, and get the absolute value of the difference between the two.

        def find_max_value(my_dict):
            max_value = float('-inf')
            key_of_max_value = None
            
            for key, value in my_dict.items():
                if value > max_value or (value == max_value):
                    max_value = value
                    key_of_max_value = key
            
            return max_value
    
        levels,predecessors = build_graph_and_traverse(connectivities_tree)
        indicator_levels = find_max_value(levels)
        while indicator_levels >1:
            keys_with_indicator = [key for key, value in levels.items() if value == indicator_levels]
            exchange = False
            for node in keys_with_indicator:
                pred = predecessors[node]
                pred_pred = predecessors[pred]
                connectivity1 = self.connectivities[node,pred]
                connectivity2 = self.connectivities[node,pred_pred]
                connectivity3 = self.connectivities[pred,pred_pred]
                min_connectivity = min(abs(connectivity2-connectivity1)/connectivity1,abs(connectivity2-connectivity3)/connectivity3)
                if min_connectivity < self._threshold:
                    exchange = True
                    min_node = min(node,pred)
                    max_node = max(node,pred)
                    connectivities_tree[min_node, max_node] = 0
                    min_node = min(node,pred_pred)
                    max_node = max(node,pred_pred)
                    connectivities_tree[min_node,max_node] = inverse_connectivities[min_node,max_node]
            if exchange == False:
                indicator_levels -= 1
            elif exchange == True:
                levels,predecessors = build_graph_and_traverse(connectivities_tree)
                indicator_levels = find_max_value(levels)

        connectivities_tree_indices = [
            connectivities_tree[i].nonzero()[1]
            for i in range(connectivities_tree.shape[0])
        ]
        connectivities_tree = sp.sparse.lil_matrix(
            self.connectivities.shape, dtype=float
        )
        for i, neighbors in enumerate(connectivities_tree_indices):
            if len(neighbors) > 0:
                connectivities_tree[i, neighbors] = self.connectivities[i, neighbors]
        return connectivities_tree.tocsr()
