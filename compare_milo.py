import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
import os

# Generate synthetic dataset
np.random.seed(42)
n_cells = 3000
n_genes = 200
n_samples = 6

print("Generating synthetic data...")
# Base counts
counts = np.random.poisson(lam=1.5, size=(n_cells, n_genes))
X = csr_matrix(counts)

samples = np.random.choice([f"Sample_{i}" for i in range(n_samples)], size=n_cells)
conditions = np.array(["ConditionA" if int(s.split("_")[1]) < 3 else "ConditionB" for s in samples])

obs = pd.DataFrame({
    "sample": samples,
    "condition": conditions
})

adata = sc.AnnData(X=X, obs=obs)

# Let's add some artificial DA effect by modifying PCA coords for ConditionB
sc.pp.pca(adata, n_comps=30)

# Cells in ConditionB have a slight shift in PCA space to create DA
is_condB = adata.obs['condition'] == 'ConditionB'
adata.obsm['X_pca'][is_condB, 0] += 2.0

sc.pp.neighbors(adata, n_neighbors=30)
sc.tl.umap(adata)

print("Running milorpy pipeline...")
import milorpy
t0 = time.time()
milorpy.build_graph(adata, k=30, d=30)
milorpy.make_nhoods(adata, prop=0.1, k=30, d=30, random_state=42)
milorpy.count_cells(adata, sample_col='sample')
milorpy.calc_nhood_distance(adata, d=30)
design_df = pd.DataFrame({'condition': ['ConditionA', 'ConditionA', 'ConditionA', 'ConditionB', 'ConditionB', 'ConditionB']}, 
                         index=[f"Sample_{i}" for i in range(6)])
milorpy.test_nhoods(adata, design='~ condition', design_df=design_df)
py_time = time.time() - t0
print(f"milorpy took {py_time:.2f}s")

py_res = adata.uns['nhood_test_results']

print("Running miloR pipeline (via rpy2)...")
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
import anndata2ri
from rpy2.robjects.packages import importr

milor = importr('miloR')
scater = importr('scater')
base = importr('base')

adata_r = adata.copy()
adata_r.uns = {}  # Clear uns to avoid anndata2ri conversion errors on nested dicts from scanpy PCA
t0 = time.time()
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
''')
r_time = time.time() - t0
print(f"miloR took {r_time:.2f}s")

with localconverter(default_converter + pandas2ri.converter):
    r_res = ro.conversion.rpy2py(ro.globalenv['res'])

print("Plotting comparison...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Py Results
axes[0].hist(py_res['logFC'], bins=30, alpha=0.7, color='blue', label='milorpy')
axes[0].set_title('milorpy logFC Distribution')
axes[0].set_xlabel('logFC')
axes[0].set_ylabel('Frequency')

# Plot R Results
axes[1].hist(r_res['logFC'], bins=30, alpha=0.7, color='red', label='miloR')
axes[1].set_title('miloR logFC Distribution')
axes[1].set_xlabel('logFC')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('/Users/halu/Code/milo/milo_comparison.png')
print("Saved comparison plot to /Users/halu/Code/milo/milo_comparison.png")
