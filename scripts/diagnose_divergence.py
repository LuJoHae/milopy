"""
Diagnostic script: Force the SAME neighborhoods into both pipelines
to isolate whether divergence comes from:
  (A) neighborhood sampling differences, or
  (B) downstream logic (count_cells, calc_nhood_distance, test_nhoods)
"""
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import pearsonr
import time

# ---- Generate synthetic dataset (identical to before) ----
np.random.seed(42)
n_cells = 3000
n_genes = 200
n_samples = 6

print("=== DIAGNOSTIC: Isolating sources of divergence ===\n")

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

# ---- Step 1: Run miloR to get its neighborhoods ----
print("Step 1: Running miloR to extract its neighborhoods...")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
import anndata2ri
from rpy2.robjects.packages import importr

milor = importr('miloR')
base = importr('base')
matrix_pkg = importr('Matrix')

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

# Extract R nhood matrix as dense
r_nhood_mat <- as.matrix(nhoods(milo))
r_nhood_counts <- nhoodCounts(milo)
r_nhood_counts_dense <- as.matrix(r_nhood_counts)

# Extract nhood distances
r_nhood_dists <- as.matrix(nhoodDistances(milo))
''')

# Pull R results into Python
with localconverter(default_converter + pandas2ri.converter):
    r_test_res = ro.conversion.rpy2py(ro.globalenv['r_test_res'])

with localconverter(default_converter + numpy2ri.converter):
    r_nhood_mat = ro.conversion.rpy2py(ro.globalenv['r_nhood_mat'])
    r_nhood_counts_dense = ro.conversion.rpy2py(ro.globalenv['r_nhood_counts_dense'])

r_n_nhoods = r_nhood_mat.shape[1]
print(f"  R produced {r_n_nhoods} neighborhoods")
print(f"  R nhood_mat shape: {r_nhood_mat.shape}")
print(f"  R nhood_counts shape: {r_nhood_counts_dense.shape}")
print(f"  R test_res shape: {r_test_res.shape}")
print(f"  R test_res columns: {list(r_test_res.columns)}")
print(f"  R test_res head:\n{r_test_res.head()}\n")

# ---- Step 2: Run milopy on same data ----
print("Step 2: Running milopy pipeline...")
import milopy
milopy.build_graph(adata, k=30, d=30)
milopy.make_nhoods(adata, prop=0.1, k=30, d=30, random_state=42)
milopy.count_cells(adata, sample_col='sample')
milopy.calc_nhood_distance(adata, d=30)
milopy.test_nhoods(adata, design='~ condition', design_df=design_df)

py_res = adata.uns['nhood_test_results']
py_nhood_mat = adata.obsm['nhoods']
if hasattr(py_nhood_mat, 'toarray'):
    py_nhood_mat_dense = py_nhood_mat.toarray()
else:
    py_nhood_mat_dense = py_nhood_mat
py_nhood_counts = adata.uns['nhood_counts']

py_n_nhoods = py_nhood_mat_dense.shape[1]
print(f"  Py produced {py_n_nhoods} neighborhoods")
print(f"  Py test_res columns: {list(py_res.columns)}")
print(f"  Py test_res head:\n{py_res.head()}\n")

# ---- Step 3: Diagnose - Force R neighborhoods into Python pipeline ----
print("Step 3: Force R neighborhoods into Python pipeline and rerun downstream...")

adata_forced = adata.copy()
# Set the nhoods matrix to be the R nhoods
adata_forced.obsm['nhoods'] = csc_matrix(r_nhood_mat)
# Derive nhood indices from the R nhood matrix (index cells)
r_nhood_indices = []
for j in range(r_nhood_mat.shape[1]):
    col = r_nhood_mat[:, j]
    members = np.where(col > 0)[0]
    r_nhood_indices.append(members[0])  # placeholder
r_nhood_indices = np.array(r_nhood_indices)
adata_forced.uns['nhood_indices'] = r_nhood_indices

# Rerun count_cells
milopy.count_cells(adata_forced, sample_col='sample')
py_forced_counts = adata_forced.uns['nhood_counts']

print(f"  Forced Py nhood_counts shape: {py_forced_counts.shape}")
print(f"  R nhood_counts shape: {r_nhood_counts_dense.shape}")

# Compare count matrices
# The R count matrix has samples as rows, nhoods as cols (or vice versa)
# Let's check
print(f"\n  R counts sample (first 5x5):\n{r_nhood_counts_dense[:5, :5]}")
print(f"  Py forced counts sample (first 5 rows):\n{py_forced_counts.head()}")

# Try to align: R nhood_counts might be samples x nhoods or nhoods x samples
# Our count_cells produces nhoods x samples
r_counts_df = pd.DataFrame(r_nhood_counts_dense)
print(f"\n  R counts shape: {r_counts_df.shape}")
print(f"  Py counts shape: {py_forced_counts.shape}")

# If R is samples x nhoods, transpose it
if r_counts_df.shape[0] == n_samples:
    print("  R counts appears to be samples x nhoods, transposing...")
    r_counts_df = r_counts_df.T

# Now both should be nhoods x samples
print(f"  Aligned R counts shape: {r_counts_df.shape}")
print(f"  Aligned Py counts shape: {py_forced_counts.shape}")

# Compare element-by-element
r_counts_vals = r_counts_df.values.flatten().astype(float)
py_counts_vals = py_forced_counts.values.flatten().astype(float)

if len(r_counts_vals) == len(py_counts_vals):
    count_corr, count_p = pearsonr(r_counts_vals, py_counts_vals)
    count_exact = np.allclose(r_counts_vals, py_counts_vals)
    print(f"\n  count_cells comparison (forced same nhoods):")
    print(f"    Pearson r = {count_corr:.6f}")
    print(f"    Exact match: {count_exact}")
    print(f"    Max abs diff: {np.max(np.abs(r_counts_vals - py_counts_vals))}")
else:
    print(f"  Cannot compare: different sizes {len(r_counts_vals)} vs {len(py_counts_vals)}")

# ---- Step 4: Run test_nhoods on the forced counts ----
print("\nStep 4: Running test_nhoods with forced R neighborhoods...")
milopy.calc_nhood_distance(adata_forced, d=30)
milopy.test_nhoods(adata_forced, design='~ condition', design_df=design_df)
py_forced_res = adata_forced.uns['nhood_test_results']

print(f"  Py forced test_res shape: {py_forced_res.shape}")
print(f"  R test_res shape: {r_test_res.shape}")

# Compare all 5 metrics
metrics = ['logFC', 'logCPM', 'F', 'PValue', 'FDR']
print(f"\n=== FINAL: Correlations with FORCED same neighborhoods ===")
for metric in metrics:
    if metric in py_forced_res.columns and metric in r_test_res.columns:
        py_vals = py_forced_res[metric].values
        r_vals = r_test_res[metric].values
        if len(py_vals) == len(r_vals):
            corr, pval = pearsonr(py_vals, r_vals)
            exact = np.allclose(py_vals, r_vals, atol=1e-10)
            max_diff = np.max(np.abs(py_vals - r_vals))
            print(f"  {metric}: r={corr:.6f}, exact={exact}, max_diff={max_diff:.2e}")
        else:
            print(f"  {metric}: size mismatch {len(py_vals)} vs {len(r_vals)}")
    else:
        print(f"  {metric}: missing from one or both results")

# Also check if the issue is the SpatialFDR
if 'SpatialFDR' in r_test_res.columns:
    print(f"\n  NOTE: R has 'SpatialFDR' column. Columns in R result: {list(r_test_res.columns)}")
