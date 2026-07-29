"""
Correlation pipeline: Force R neighborhoods into milopy (now using edgepython)
and compare all 5 outputs.
"""
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import pearsonr

# Generate synthetic dataset
np.random.seed(42)
n_cells = 3000
n_genes = 200
n_samples = 6

print("=== Correlation Pipeline: milopy (edgepython) vs miloR (R) ===\n")

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

# ---- Run miloR in R ----
print("Running miloR (R)...")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
import anndata2ri
from rpy2.robjects.packages import importr

milor = importr('miloR')

adata_r = adata.copy()
adata_r.uns = {}
with localconverter(default_converter + pandas2ri.converter + anndata2ri.converter):
    r_sce = ro.conversion.py2rpy(adata_r)

design_df = pd.DataFrame(
    {'condition': ['ConditionA']*3 + ['ConditionB']*3},
    index=[f"Sample_{i}" for i in range(6)]
)
with localconverter(default_converter + pandas2ri.converter):
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
r_test_res <- testNhoods(milo, design=design_mat, design.df=design_df)

# Extract R artifacts
r_nhood_mat <- as.matrix(nhoods(milo))
r_nhood_counts_dense <- as.matrix(nhoodCounts(milo))
''')

with localconverter(default_converter + pandas2ri.converter):
    r_test_res = ro.conversion.rpy2py(ro.globalenv['r_test_res'])
with localconverter(default_converter + numpy2ri.converter):
    r_nhood_mat = ro.conversion.rpy2py(ro.globalenv['r_nhood_mat'])
    r_nhood_counts_dense = ro.conversion.rpy2py(ro.globalenv['r_nhood_counts_dense'])

print(f"  R: {r_nhood_mat.shape[1]} neighborhoods")
print(f"  R test columns: {list(r_test_res.columns)}")

# ---- Force R neighborhoods into milopy ----
print("\nForcing R neighborhoods into milopy pipeline...")
import milopy

adata_forced = adata.copy()
adata_forced.obsm['nhoods'] = csc_matrix(r_nhood_mat)
adata_forced.uns['nhood_indices'] = np.array([
    np.where(r_nhood_mat[:, j] > 0)[0][0] for j in range(r_nhood_mat.shape[1])
])

milopy.count_cells(adata_forced, sample_col='sample')
milopy.calc_nhood_distance(adata_forced, d=30)
milopy.test_nhoods(adata_forced, design='~ condition', design_df=design_df)

py_res = adata_forced.uns['nhood_test_results']
print(f"  Py: {len(py_res)} neighborhoods")
print(f"  Py test columns: {list(py_res.columns)}")

# ---- Compare ----
print("\n=== Correlations (forced same neighborhoods) ===")
metrics = ['logFC', 'logCPM', 'F', 'PValue', 'FDR']
for metric in metrics:
    if metric in py_res.columns and metric in r_test_res.columns:
        py_vals = py_res[metric].values
        r_vals = r_test_res[metric].values
        if len(py_vals) == len(r_vals):
            corr, pval = pearsonr(py_vals, r_vals)
            exact = np.allclose(py_vals, r_vals, atol=1e-10)
            max_diff = np.max(np.abs(py_vals - r_vals))
            print(f"  {metric:8s}: r={corr:.6f}, exact={exact}, max_diff={max_diff:.2e}")
        else:
            print(f"  {metric}: size mismatch {len(py_vals)} vs {len(r_vals)}")

# Also show first few rows side-by-side
print("\n--- R results (first 5) ---")
print(r_test_res[['logFC','logCPM','F','PValue','FDR']].head())
print("\n--- Py results (first 5) ---")
print(py_res[['logFC','logCPM','F','PValue','FDR']].head())
