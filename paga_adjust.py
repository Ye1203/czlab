
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy as sp


def paga_adjust(
    adata: AnnData,
    groups: str | None = None,
    root: str | None = None,
    knn_n: num | None = None,
    conn_threshold: num| None = None,
    ad_space: str| None = None
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
        The number of nearest neighbors used by knn, the default is 5. It is not recommended to
        use a value that is too large.
    conn_threshold
        The threshold for correcting the connection tree, default is 0.5.
    ad_space
        Graphics space, by default, uses 'X_dif' under 'adata.obsm'. The selected space must have equal weights,
        which means that pca cannot be used.

    Returns
    -------
    

    Notes
    -----
    

    """
    if 'paga' not in adata.uns:
        raise ValueError(
                "You need to run `sc.tl.paga` first to compute a connectivities."
            )
        if "connectivities"  not in adata.uns['paga']:
            raise ValueError(
                "You need to run `sc.tl.paga` first to compute a connectivities."
            )

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
        knn_n = 5
        print('The value of knn_n is not entered, the default value is 5.')

    if conn_threshold is None:
        conn_threshold = 0.5
        print('The value of conn_threshold is not entered, the default value is 0.5.')

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


    paga_adjust = PAGA_A(adata, groups = groups, knn_n = knn_n, conn_threshold = conn_threshold, ad_space = ad_space)
    paga_adjust.compute_tree()
    adata.uns["paga"]["connectivities_tree"] = paga_adjust.connectivities_tree
    return None

class PAGA_A:
    def __init__(self, adata, groups, knn_n, conn_threshold, ad_space):
        self._adata = adata
        self._groups = groups
        self._knn_n = knn_n
        self._conn_threshold = conn_threshold
        self._ad_space = ad_space
    
    def compute_tree(self):
        self.connectivities_tree = self._adata.uns["paga"]["connectivities_tree"].copy()
        return self.connectivities_tree
