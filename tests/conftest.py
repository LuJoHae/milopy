import pytest
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import csr_matrix

@pytest.fixture
def synthetic_adata():
    # Set seed for reproducibility
    np.random.seed(42)
    
    n_cells = 1000
    n_genes = 500
    n_samples = 6
    
    # Random count matrix
    counts = np.random.poisson(lam=1.0, size=(n_cells, n_genes))
    X = csr_matrix(counts)
    
    # Cell metadata
    samples = np.random.choice([f"Sample_{i}" for i in range(n_samples)], size=n_cells)
    conditions = np.array(["ConditionA" if int(s.split("_")[1]) < 3 else "ConditionB" for s in samples])
    
    obs = pd.DataFrame({
        "sample": samples,
        "condition": conditions
    })
    
    # Mock PCA
    pca = np.random.randn(n_cells, 30)
    
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsm['X_pca'] = pca
    
    return adata
