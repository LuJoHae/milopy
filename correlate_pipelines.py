import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Generate synthetic dataset
np.random.seed(42)
n_cells = 3000
n_genes = 200
n_samples = 6

print("Generating synthetic data...")
counts = np.random.poisson(lam=1.5, size=(n_cells, n_genes))
X = csr_matrix(counts)
samples = np.random.choice([f"Sample_{i}" for i in range(n_samples)], size=n_cells)
conditions = np.array(["ConditionA" if int(s.split("_")[1]) < 3 else "ConditionB" for s in samples])
obs = pd.DataFrame({"sample": samples, "condition": conditions})
adata = sc.AnnData(X=X, obs=obs)

sc.pp.pca(adata, n_comps=30)
is_condB = adata.obs['condition'] == 'ConditionB'
adata.obsm['X_pca'][is_condB, 0] += 2.0
sc.pp.neighbors(adata, n_neighbors=30)
sc.tl.umap(adata)

print("Running milorpy pipeline...")
import milorpy
milorpy.build_graph(adata, k=30, d=30)
milorpy.make_nhoods(adata, prop=0.1, k=30, d=30, random_state=42)
milorpy.count_cells(adata, sample_col='sample')
milorpy.calc_nhood_distance(adata, d=30)
design_df = pd.DataFrame({'condition': ['ConditionA', 'ConditionA', 'ConditionA', 'ConditionB', 'ConditionB', 'ConditionB']}, 
                         index=[f"Sample_{i}" for i in range(6)])
milorpy.test_nhoods(adata, design='~ condition', design_df=design_df)
py_res = adata.uns['nhood_test_results']

# Compute py centroids
py_nhoods_mat = adata.obsm['nhoods']
if hasattr(py_nhoods_mat, "toarray"):
    py_nhoods_mat = py_nhoods_mat.toarray()
py_centroids = (py_nhoods_mat.T @ adata.obsm['X_pca']) / py_nhoods_mat.sum(axis=0)[:, None]

print("Running miloR pipeline (via rpy2)...")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
import anndata2ri
from rpy2.robjects.packages import importr

milor = importr('miloR')
adata_r = adata.copy()
adata_r.uns = {}
with localconverter(default_converter + pandas2ri.converter + anndata2ri.converter):
    r_sce = ro.conversion.py2rpy(adata_r)
    r_design_df = ro.conversion.py2rpy(design_df)

ro.globalenv['r_sce'] = r_sce
ro.globalenv['design_df'] = r_design_df

ro.r('''
library(miloR)
library(SingleCellExperiment)

milo <- Milo(r_sce)
milo <- buildGraph(milo, k=30, d=30, reduced.dim="PCA")
set.seed(42)
milo <- makeNhoods(milo, prop=0.1, k=30, d=30, refined=TRUE)
milo <- countCells(milo, meta.data=as.data.frame(colData(milo)), sample="sample")
milo <- calcNhoodDistance(milo, d=30, reduced.dim="PCA")

design_mat <- model.matrix(~ condition, data=design_df)
res <- testNhoods(milo, design=design_mat, design.df=design_df)

# Get R centroids
pca_mat <- reducedDim(milo, "PCA")
nhood_mat <- nhoods(milo)
r_centroids <- t(as.matrix(nhood_mat)) %*% pca_mat / colSums(as.matrix(nhood_mat))
''')

with localconverter(default_converter + pandas2ri.converter):
    r_res = ro.conversion.rpy2py(ro.globalenv['res'])
    r_centroids = ro.conversion.rpy2py(ro.globalenv['r_centroids'])

print(f"Number of py nhoods: {len(py_res)}")
print(f"Number of r nhoods: {len(r_res)}")

# Pair each python neighborhood to the closest R neighborhood
dists = cdist(py_centroids, r_centroids)
closest_r_idx = np.argmin(dists, axis=1)

print("\n--- Correlations between paired neighborhoods ---")
metrics = ['logFC', 'logCPM', 'F', 'PValue', 'FDR']
for metric in metrics:
    if metric in py_res.columns and metric in r_res.columns:
        py_vals = py_res[metric].values
        r_vals = r_res.iloc[closest_r_idx][metric].values
        corr, pval = pearsonr(py_vals, r_vals)
        print(f"{metric}: Pearson r = {corr:.4f} (p={pval:.2e})")
