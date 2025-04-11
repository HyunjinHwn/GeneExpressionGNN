import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import scipy.stats as stats
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
def calculate_Error(GT, INF):
  # GT / INF : (12320, 176) numpy ndarray (row, column orders should match)
  # output: list of Error values (length: 176)
  err = []
  for i in range(INF.shape[1]):
    err.append(np.sqrt(np.sum((INF[:,i]-GT[:,i])**2)))
  return err

def calculate_SCC(GT, INF):
  # GT / INF : (12320, 176) numpy ndarray (row, column orders should match)
  # output: list of SCC values (length: 176)
  from scipy.stats import spearmanr
  scc = []
  for i in range(INF.shape[1]):
    scc.append(spearmanr(GT[:,i].flatten(), INF[:,i].flatten())[0])
  return scc

def calculate_PCC(GT, INF):
  from scipy.stats import pearsonr
  pcc = []
  for i in range(INF.shape[1]):
    pcc.append(pearsonr(GT[:,i].flatten(), INF[:,i].flatten())[0])
  return pcc

def recall_for_LM(GT, INF, lmid, index_all):
    # GT, INF: (12320, 176) numpy ndarray, whose order of rows (genes) matches with that of geneid_12320.txt
    # lmid: list of landmark ids
    # outfn: output text file name where the gene ids of BING (=landmark + inferred genes with Recall>0.95) are written
    nonlmid = index_all.drop(lmid).tolist()
    lm_sep_order = lmid + nonlmid
    GT_sep = pd.DataFrame(GT, index=index_all)
    INF_sep = pd.DataFrame(INF, index=index_all)
    GT_sep = GT_sep.loc[lm_sep_order,:].to_numpy()
    INF_sep = INF_sep.loc[lm_sep_order,:].to_numpy()
    r95, r99, BING_df = recall_test(GT_sep, INF_sep, lmid=lmid, nonlmid=nonlmid)
    return r95, r99, BING_df

def recall_test(GT, INF, lmid, nonlmid):
    # Load gene information for annotation
    gene_info = pd.read_csv(os.path.join(current_dir, "metadata/geneinfo_beta.txt"), sep="\t", dtype={"gene_id": str})
    gene_info_dict = gene_info.set_index("gene_id")["gene_symbol"].to_dict()

    GT_rank = stats.rankdata(GT, axis=1)
    INF_rank = stats.rankdata(INF, axis=1)

    recallcorr = np.corrcoef(GT_rank, INF_rank)[:12320, 12320:]
    self_corr = np.diagonal(recallcorr).tolist()

    np.fill_diagonal(recallcorr, np.nan)
    recallcorr = recallcorr[~np.isnan(recallcorr)].tolist()
    null_dist = recallcorr

    def calc_recall(null_dist, self_corr, qval):
        r_above_threshold = []
        threshold = np.percentile(null_dist, q=qval)
        self_corr = np.array(self_corr)
        r_above_threshold = (self_corr>=threshold).nonzero()[0].tolist()
        return r_above_threshold, len(r_above_threshold), threshold

    r95_list, r95, r95_threshold = calc_recall(null_dist, self_corr, 95)
    r99_list, r99, r99_threshold = calc_recall(null_dist, self_corr, 99)

    null_dist_np = np.sort(null_dist)
    positions = np.searchsorted(null_dist_np, self_corr, side='right')
    percentiles = (positions / len(null_dist_np)) * 100

    all_gene_symbols = [gene_info_dict.get(str(gene), str(gene)) for gene in lmid + nonlmid]
    all_self_corr = self_corr
    all_r95_flags = [1 if i in r95_list else 0 for i in range(len(lmid + nonlmid))]
    all_r99_flags = [1 if i in r99_list else 0 for i in range(len(lmid + nonlmid))]

    BING_df = pd.DataFrame({
        "Gene_Symbol": all_gene_symbols,
        "Gene_ID": lmid + nonlmid,
        "Self_Correlation": all_self_corr,
        "Recall": percentiles,
        "Recall_95": all_r95_flags,
        "Recall_99": all_r99_flags
    })
    return r95, r99, BING_df

def level3_to_level4(INF, rid, estimate=False):
    def rzs(mat): # calculates robust z-score
        medians = mat.median(axis=1)
        sub = mat.subtract(medians, axis='index')
        mads = abs(sub).median(axis=1)
        # if estimate option is used, change the min_mad to the 1st percentile of mads if the value is larger than 0.01
        if estimate:
            min_mad = max(np.nanpercentile(mads, 1), 0.01)
        else:
            min_mad = 0.01
        mads = mads.clip(lower=min_mad)
        zscore_df = sub.divide(mads * 1.4826, axis='index')
        return zscore_df

    df = pd.DataFrame(INF, index=rid)
    df = rzs(df)
    return df