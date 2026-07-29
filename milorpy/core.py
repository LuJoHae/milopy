import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata

def build_graph(adata, k=30, d=30, **kwargs):
    """
    Builds a kNN graph. Wraps scanpy.pp.neighbors.
    """
    if 'X_pca' not in adata.obsm:
        raise ValueError("PCA must be computed first. Run sc.pp.pca(adata)")
        
    sc.pp.neighbors(adata, n_neighbors=k, n_pcs=d, **kwargs)
    return adata

def make_nhoods(adata, prop=0.1, k=30, d=30, refined=True, random_state=None):
    """
    Defines neighbourhoods on the kNN graph.
    """
    if 'distances' not in adata.obsp:
        raise ValueError("kNN graph not found. Run build_graph(adata)")
        
    n_cells = adata.n_obs
    n_nhoods = int(np.round(n_cells * prop))
    
    if random_state is not None:
        np.random.seed(random_state)
        
    # Initial random sampling
    vertex_indices = np.random.choice(n_cells, size=n_nhoods, replace=False)
    
    knn_graph = adata.obsp['connectivities']
    
    if refined:
        # Refinement step: for each random vertex, find its neighbourhood,
        # compute the median profile in PCA space, and pick the closest vertex.
        pca_coords = adata.obsm['X_pca'][:, :d]
        refined_vertices = []
        for v in vertex_indices:
            # Find neighbors of v
            neighbors = knn_graph[v].nonzero()[1]
            # include self
            neighborhood = np.append(neighbors, v)
            
            # Compute median profile
            median_profile = np.median(pca_coords[neighborhood], axis=0)
            
            # Find vertex in neighborhood closest to median
            distances = np.linalg.norm(pca_coords[neighborhood] - median_profile, axis=1)
            closest_idx = neighborhood[np.argmin(distances)]
            refined_vertices.append(closest_idx)
            
        # Remove duplicates
        vertex_indices = np.unique(refined_vertices)
        
    # Store neighborhoods as a sparse matrix: cells x nhoods
    rows = []
    cols = []
    
    for i, v in enumerate(vertex_indices):
        neighbors = knn_graph[v].nonzero()[1]
        neighborhood = np.append(neighbors, v)
        rows.extend(neighborhood)
        cols.extend([i] * len(neighborhood))
        
    nhoods_mat = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, len(vertex_indices))).tocsc()
    
    adata.obsm['nhoods'] = nhoods_mat
    adata.uns['nhood_indices'] = vertex_indices
    return adata

def count_cells(adata, sample_col):
    """
    Counts cells in each neighbourhood across samples.
    """
    if 'nhoods' not in adata.obsm:
        raise ValueError("Neighborhoods not found. Run make_nhoods(adata)")
        
    if sample_col not in adata.obs.columns:
        raise ValueError(f"Sample column '{sample_col}' not found in adata.obs")
        
    samples = adata.obs[sample_col].astype('category')
    sample_categories = samples.cat.categories
    sample_codes = samples.cat.codes.values
    
    n_nhoods = adata.obsm['nhoods'].shape[1]
    n_samples = len(sample_categories)
    
    # Initialize count matrix
    counts = np.zeros((n_nhoods, n_samples))
    
    nhoods_mat = adata.obsm['nhoods'].tocsc()
    
    for i in range(n_nhoods):
        cells_in_nhood = nhoods_mat[:, i].nonzero()[0]
        # Count cells per sample in this neighborhood
        nhood_sample_codes = sample_codes[cells_in_nhood]
        counts[i, :] = np.bincount(nhood_sample_codes, minlength=n_samples)
        
    count_df = pd.DataFrame(counts, columns=sample_categories)
    
    # store in uns
    adata.uns['nhood_counts'] = count_df
    return adata

def calc_nhood_distance(adata, d=30):
    """
    Calculates distances between neighbourhoods based on overlap or PCA distance.
    """
    if 'nhoods' not in adata.obsm:
        raise ValueError("Neighborhoods not found. Run make_nhoods(adata)")
        
    nhoods_mat = adata.obsm['nhoods'].tocsc()
    pca_coords = adata.obsm['X_pca'][:, :d]
    
    n_nhoods = nhoods_mat.shape[1]
    
    # compute median of each neighborhood
    nhood_medians = np.zeros((n_nhoods, d))
    for i in range(n_nhoods):
        cells = nhoods_mat[:, i].nonzero()[0]
        nhood_medians[i, :] = np.median(pca_coords[cells, :], axis=0)
        
    # compute euclidean distance between medians
    from scipy.spatial.distance import pdist, squareform
    dists = pdist(nhood_medians, metric='euclidean')
    dist_mat = squareform(dists)
    
    adata.uns['nhood_distances'] = dist_mat
    return adata

def test_nhoods(adata, design, design_df, model_contrasts=None):
    """
    Tests for differential abundance using edgeR via rpy2.
    `design` is a formula string like '~ condition'
    `design_df` is a pandas DataFrame with row names matching sample columns in `nhood_counts`
    """
    if 'nhood_counts' not in adata.uns:
        raise ValueError("Neighborhood counts not found. Run count_cells(adata, sample_col)")
        
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter
    except Exception as e:
        raise ImportError(f"rpy2 is required for test_nhoods. Failed to import: {e}")
        
    edger = importr('edgeR')
    stats = importr('stats')
    base = importr('base')
    
    # Counts are nhoods x samples
    counts_df = adata.uns['nhood_counts']
    
    # Ensure design_df is aligned with counts_df columns
    design_df = design_df.loc[counts_df.columns]
    
    # Convert to R objects
    with localconverter(default_converter + pandas2ri.converter):
        r_counts = ro.conversion.py2rpy(counts_df)
        r_design_df = ro.conversion.py2rpy(design_df)
    
    # Create DGEList
    dge = edger.DGEList(counts=r_counts)
    dge = edger.calcNormFactors(dge, method="TMM")
    
    # Create design matrix
    formula = stats.as_formula(design)
    design_mat = stats.model_matrix(formula, data=r_design_df)
    
    # Estimate dispersion
    dge = edger.estimateDisp(dge, design_mat)
    
    # Fit QL GLM
    fit = edger.glmQLFit(dge, design_mat, robust=True)
    
    # Test
    if model_contrasts is not None:
        # e.g., model_contrasts = "conditionB - conditionA"
        make_contrasts = importr('limma').makeContrasts
        contrast_mat = make_contrasts(model_contrasts, levels=design_mat)
        res = edger.glmQLFTest(fit, contrast=contrast_mat)
    else:
        # Default to testing the last coefficient
        res = edger.glmQLFTest(fit, coef=design_mat.ncol)
        
    # Get results table
    top_tags = edger.topTags(res, n=counts_df.shape[0], sort_by="none")
    with localconverter(default_converter + pandas2ri.converter):
        res_df = ro.conversion.rpy2py(top_tags.rx2('table'))
    
    res_df.index = counts_df.index
    adata.uns['nhood_test_results'] = res_df
    return adata
