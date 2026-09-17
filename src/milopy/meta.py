import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

def test_nhoods_meta(adata, design, design_df, dataset_col, model_contrasts=None):
    """
    Random-Effects Meta-Analysis for multi-dataset Differential Abundance.
    Runs edgepython GLM independently for each dataset and combines effects using
    Inverse-Variance DerSimonian-Laird meta-analysis.
    """
    if 'nhood_counts' not in adata.uns:
        raise ValueError("Neighborhood counts not found. Run count_cells(adata, sample_col)")
        
    import edgepython as ep
    import patsy
    
    counts_df = adata.uns['nhood_counts']
    design_df = design_df.loc[counts_df.columns]
    
    if dataset_col not in design_df.columns:
        raise ValueError(f"Dataset column '{dataset_col}' not found in design_df")
        
    datasets = design_df[dataset_col].unique()
    
    n_nhoods = counts_df.shape[0]
    
    # Store effects and standard errors
    all_logfc = np.zeros((n_nhoods, len(datasets)))
    all_se = np.zeros((n_nhoods, len(datasets)))
    all_logcpm = np.zeros((n_nhoods, len(datasets)))
    
    for k, ds in enumerate(datasets):
        ds_samples = design_df[design_df[dataset_col] == ds].index
        ds_counts = counts_df[ds_samples].values.astype(float)
        ds_design = design_df.loc[ds_samples]
        
        y = ep.make_dgelist(counts=ds_counts)
        y = ep.calc_norm_factors(y, method="TMM")
        
        design_mat = patsy.dmatrix(design, ds_design, return_type='dataframe')
        design_array = np.asarray(design_mat, dtype=float)
        
        y = ep.estimate_disp(y, design=design_array)
        fit = ep.glm_ql_fit(y, design=design_array, robust=True)
        
        if model_contrasts is not None:
            res = ep.glm_ql_ftest(fit, contrast=model_contrasts)
        else:
            n_coefs = design_array.shape[1]
            res = ep.glm_ql_ftest(fit, coef=n_coefs - 1)
            
        top = ep.top_tags(res, n=n_nhoods, sort_by="none")
        res_df = pd.DataFrame(top['table'])
        
        # logFC and F-statistic from edgeR/edgepython
        logfc = res_df['logFC'].values
        f_stat = res_df['F'].values
        # standard error from F-stat (assuming 1 DF: F = (beta/se)^2 => se = |beta| / sqrt(F))
        se = np.abs(logfc) / np.sqrt(f_stat + 1e-12) # add small epsilon to avoid div by zero
        
        all_logfc[:, k] = logfc
        all_se[:, k] = se
        all_logcpm[:, k] = res_df['logCPM'].values

    # DerSimonian-Laird Random Effects Meta-Analysis
    variances = all_se**2
    w_fixed = 1.0 / variances
    w_sum = w_fixed.sum(axis=1)
    beta_fixed = (w_fixed * all_logfc).sum(axis=1) / w_sum
    
    K = len(datasets)
    Q = (w_fixed * (all_logfc - beta_fixed[:, None])**2).sum(axis=1)
    c = w_sum - (w_fixed**2).sum(axis=1) / w_sum
    tau2 = np.maximum(0, (Q - (K - 1)) / (c + 1e-12))
    
    w_random = 1.0 / (variances + tau2[:, None])
    w_random_sum = w_random.sum(axis=1)
    
    meta_logfc = (w_random * all_logfc).sum(axis=1) / w_random_sum
    meta_se = np.sqrt(1.0 / w_random_sum)
    
    z_stat = meta_logfc / meta_se
    pvals = 2 * scipy.stats.norm.sf(np.abs(z_stat))
    _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
    
    meta_res = pd.DataFrame({
        'logFC': meta_logfc,
        'SE': meta_se,
        'logCPM': np.mean(all_logcpm, axis=1),
        'PValue': pvals,
        'FDR': fdr,
        'Tau2': tau2
    }, index=counts_df.index)
    
    adata.uns['nhood_test_results'] = meta_res
    return adata

def test_nhoods_mixed(adata, design, design_df, dataset_col, model_contrasts=None):
    """
    Negative Binomial Mixed-Effects model for multi-dataset DA.
    Uses edgepython's NEBULA-LN extension (ep.glm_sc_fit) modeling
    'dataset_col' as the random effect grouping variable.
    """
    if 'nhood_counts' not in adata.uns:
        raise ValueError("Neighborhood counts not found. Run count_cells(adata, sample_col)")
        
    import edgepython as ep
    import patsy
    
    counts_df = adata.uns['nhood_counts']
    design_df = design_df.loc[counts_df.columns]
    
    if dataset_col not in design_df.columns:
        raise ValueError(f"Dataset column '{dataset_col}' not found in design_df")
        
    counts_matrix = counts_df.values.astype(float)
    
    design_mat = patsy.dmatrix(design, design_df, return_type='dataframe')
    design_array = np.asarray(design_mat, dtype=float)
    
    try:
        # Fit the NEBULA-LN model with the random effect (dataset_col)
        fit = ep.glm_sc_fit(
            y=counts_matrix,
            cell_meta=design_df,
            design=design_array,
            sample=dataset_col,
            norm_method='TMM'
        )
    except AttributeError:
        raise NotImplementedError("ep.glm_sc_fit (NEBULA-LN) is missing or not compiled in this edgepython version.")
    
    n_nhoods = counts_matrix.shape[0]
    
    if model_contrasts is not None:
        res = ep.glm_ql_ftest(fit, contrast=model_contrasts)
    else:
        n_coefs = design_array.shape[1]
        res = ep.glm_ql_ftest(fit, coef=n_coefs - 1)
        
    top = ep.top_tags(res, n=n_nhoods, sort_by="none")
    res_df = pd.DataFrame(top['table'])
    res_df.index = counts_df.index
    
    adata.uns['nhood_test_results'] = res_df
    return adata
