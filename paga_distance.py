from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix,lil_matrix

def paga_adjust(
    adata: AnnData,
    groups: Optional[str] = None,
    root: Optional[str] = None,
    knn_n: Optional[int] = None,
    ad_space: Optional[str] = None,
    center: Optional[str] = None,
    bc: Optional[float] = None,
    da_weight:Optional[float] = None
) -> AnnData | None:
    """\
    

    Parameters
    ----------
    adata
        Required. An annotated data matrix.
    groups
        Cell type. Like 'label' or 'leiden'. By default, the 'label' or 'labels' in
        'adata.obs' will be searched, and if they exist, the corresponding value will be specified.
    root
        Required. Start cell of analysis. It needs to be consistent with the input of groups. 
        In other words, the root input must appear in the 'groups'
    knn_n
        The number of nearest neighbors used by knn, the default is 10. It is not recommended to
        use a value that is too large.
    ad_space
        Graphics space, by default, uses 'X_dif' under 'adata.obsm'. The selected space must have equal weights,
        which means that pca cannot be used.
    center
        How to calculate the center of each groups in 'ad_spave'. Must be either 'mean' or 'median'. Default is 'median'.
    bc
        Bifurcation coefficient. Default is 1.1, must larger than 1. The closer to 1, the easier it is to fork when drawing a tree diagram.
    da_weight
        Distance Angle Weight. Default is 1. The value must be greater than 0. The smaller the angle, the greater the impact.
        The larger the distance, the greater the impact.

    Returns
    -------
    The output will build a ['connectivities'] and ['connectivities_tree'] under .uns['paga'] for subsequent drawing using sc.pl.paga
    
    

    """
    

    if groups is None:
        for k in ("labels", "label"):
            if k in adata.obs.columns:
                groups = k
                break
    if groups is None:
        raise ValueError(
            f"{groups} not in adata.obs."
        )

    if root is None:
        raise ValueError(
            "You must input root."
        )
    elif root is not None and root not in adata.obs[groups].values:
        raise ValueError(f"The root value '{root}' is not found in the specified group '{groups}'.")

    if knn_n is None:
        knn_n = 10
        print('The value of knn_n is not entered, the default value is 10.')

    if ad_space is None:
        if 'X_dif' in adata.obsm:
            ad_space = 'X_dif'
    else:
        if ad_space not in adata.obsm:
            raise ValueError(
                f"cannot find {ad_space} in adata.obsm."
            )
    if ad_space is None:
        raise ValueError (
            "You must enter the value of ad_space."
        )
    if center is None:
        center = "median"
    if center not in ['median','mean']:
        raise ValueError(
            "Center must be either the mean or the median"
        )
    if bc is None:
        bc = 1.1
        print('The value of bc is not entered, the default value is 1.1.')
    if da_weight is None:
        da_weight = 1.0
        print('The value of da_weight is not entered, the default value is 1.0.')

    


    paga_distance = PAGA_A(adata, groups = groups, root = root, knn_n = knn_n, ad_space = ad_space, center = center, bc = bc, da_weight = da_weight)
    paga_distance._compute_connectivities()
    if "paga" not in adata.uns:
        adata.uns["paga"] = {}
    adata.uns["paga"]["connectivities"] = paga_distance.connectivities
    adata.uns["paga"]["connectivities_tree"] = paga_distance.connectivities_tree
    adata.uns["paga"]["groups"] = groups
    return None

class PAGA_A:
    def __init__(self, adata, groups, root, knn_n,  ad_space, center, bc, da_weight):
        self._adata = adata
        self._groups = groups
        self._root = root
        self._knn_n = knn_n
        self._ad_space = ad_space
        self._center = center
        self._bc = bc
        self._da_weight = da_weight
    
    def _compute_connectivities(self):
        data_clc = pd.DataFrame({
            'cell_id': self._adata.obs_names,
            'groups': self._adata.obs[self._groups].values,
            'coordinate': list(self._adata.obsm[self._ad_space].copy())
        })
        # Calculate group center points
        def find_center(data, method='median'):
            if method == 'median':
                center_coordinate = data.groupby('groups', observed=True)['coordinate'].apply(lambda x: np.median(np.vstack(x), axis=0)).reset_index()
            elif method == 'mean':
                center_coordinate = data.groupby('groups', observed=True)['coordinate'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
            return center_coordinate
        # Find the nearest non-group knn's direction

        def find_knn_direction(data, k=10):

            
            group_dict = {}
            for group in data['groups'].unique():
                group_dict[group] = data[data['groups'] != group]

            
            direction_results = []

            def piecewise_function(x):
                return np.where(x < 0, -1 / (x**2 + 1), 1 / (x**2 + 1))

            for i, row in data.iterrows():
                current_group = row['groups']
                other_group_data = group_dict[current_group]
                other_coordinates = np.vstack(other_group_data['coordinate'].values)
                current_coordinate = np.array(row['coordinate']).reshape(1, -1)
                distances = np.linalg.norm(other_coordinates - current_coordinate, axis=1)
                knn_indices = np.argsort(distances)[:k]
                vectors = other_coordinates[knn_indices] - current_coordinate
                transformed_vectors = piecewise_function(vectors)
                direction_sum = np.sum(transformed_vectors, axis=0)
                direction_results.append(direction_sum)

            data['direction'] = direction_results

            return data

        
        def projection_length(vector, direction, da_weight=1):
            dot_product = np.dot(vector, direction)
            vector_norm = np.linalg.norm(vector)
            direction_norm = np.linalg.norm(direction)
            cos_theta = dot_product / (vector_norm * direction_norm)
            projection = 1/vector_norm ** da_weight * (cos_theta+1) #control distance and angle 
            
            return projection

        def compute_projections(data, centers):
            for group, center in zip(centers['groups'], centers['coordinate']):
                projections = []
                for idx, row in data.iterrows():
                    vector = np.array(center) - np.array(row['coordinate'])
                    proj_length = projection_length(vector, np.array(row['direction']), self._da_weight)
                    projections.append(proj_length)
                
                data[group] = projections
        
        def rank_projections(data,bc = 1.1):
            group_length = len(center_coordinate['groups'])
            for idx, row in data.iterrows():
                projections = row[4:4+group_length].values  
                rankings = {}
                for col in data.columns[4:4+group_length]:
                    group = col
                    if row['groups'] == group:
                        rankings[group + '_rank'] = 0
                    else:
                        sorted_projs = np.sort(projections)  
                        rank = np.where(sorted_projs == row[col])[0][0]  
                        rankings[group + '_rank'] = rank
                
                for key, value in rankings.items():
                    if value !=0:
                        data.loc[idx, key] = float(bc)**(value-round(group_length*0.75))
            proj_columns = data.columns[4:4+group_length]
            data.drop(columns=proj_columns, inplace=True)
            group_means = data.groupby('groups',observed=False)[proj_columns+'_rank'].mean().fillna(0).to_numpy()

            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            flattened = group_means.flatten()
            scaled_flattened = min_max_scaler.fit_transform(flattened.reshape(-1, 1)).ravel()
            group_means = scaled_flattened.reshape(group_means.shape)
            return group_means
        
                

        center_coordinate = find_center(data_clc, method=self._center)
        
        data_clc = find_knn_direction(data_clc, k=self._knn_n)

        compute_projections(data_clc, center_coordinate)

        

        self.connectivities = csr_matrix(rank_projections(data_clc, bc = self._bc))

        self.connectivities_tree = self._compute_connectivities_tree()
        return None

    def _compute_connectivities_tree(self):
        unique_groups = sorted(self._adata.obs[self._groups].unique())
        if self._root in unique_groups:
            self._root_index = unique_groups.index(self._root)
        else:
            raise ValueError(f"{self._root} not in the {self._groups_key}")        
        connectivities = self.connectivities.copy()
        from_list = [self._root_index]
        connectivities_tree = lil_matrix(connectivities.shape, dtype=float)

        connectivities[:, self._root_index] = 0

        while connectivities.count_nonzero() > 0:
            rows, cols = connectivities.nonzero()
            candidate_edges = [(r, c, connectivities[r, c]) for r, c in zip(rows, cols) if r in from_list]

            if not candidate_edges:
                break
            find_from, find_to, max_value = max(candidate_edges, key=lambda x: x[2])
            connectivities_tree[find_from, find_to] = max_value
            from_list.append(find_to)
            connectivities[:, find_to] = 0

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

        # from scipy.sparse.csgraph import minimum_spanning_tree     
        # inverse_connectivities = self.connectivities.copy()
        # inverse_connectivities.data = 1.0 / inverse_connectivities.data
        # connectivities_tree = minimum_spanning_tree(inverse_connectivities)
        # connectivities_tree_indices = [
        #     connectivities_tree[i].nonzero()[1]
        #     for i in range(connectivities_tree.shape[0])
        # ]
        # connectivities_tree = sp.sparse.lil_matrix(
        #     self.connectivities.shape, dtype=float
        # )
        # for i, neighbors in enumerate(connectivities_tree_indices):
        #     if len(neighbors) > 0:
        #         connectivities_tree[i, neighbors] = self.connectivities[i, neighbors]
        # return connectivities_tree.tocsr()
