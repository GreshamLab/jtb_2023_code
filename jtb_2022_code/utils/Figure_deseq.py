# Inspired by https://github.com/wckdouglas/diffexpr
import rpy2.robjects as _robjects
from rpy2.robjects import pandas2ri as _pandas2ri, numpy2ri as _numpy2ri, Formula as _Formula
from rpy2.robjects.conversion import localconverter as _localconverter
from rpy2.robjects.packages import importr as _importr
from rpy2.rinterface_lib import sexp as _sexp
import pandas as _pd
import numpy as _np

def _numpy_to_r(p_obj):
    with _localconverter(_robjects.default_converter + _numpy2ri.converter):
        return _robjects.conversion.py2rpy(p_obj)

def _pandas_to_r(p_obj, as_matrix=False, transpose=False):
    if not as_matrix:
        with _localconverter(_robjects.default_converter + _pandas2ri.converter):
            return _robjects.conversion.py2rpy(p_obj if not transpose else p_obj.T)
    else:
        with _localconverter(_robjects.default_converter + _numpy2ri.converter):
            mat = _robjects.conversion.py2rpy(p_obj.values if not transpose else p_obj.values.T)
            r = p_obj.index.astype(str).tolist()
            c = p_obj.columns.astype(str).tolist()
            mat.rownames = _robjects.StrVector(r if not transpose else c)
            mat.colnames = _robjects.StrVector(c if not transpose else r)
            return mat
            
def _r_to_numpy(r_obj):
    with _localconverter(_robjects.default_converter + _numpy2ri.converter):
        return _robjects.conversion.rpy2py(r_obj)    

def _r_to_pandas(r_obj):
    with _localconverter(_robjects.default_converter + _pandas2ri.converter):
        return _robjects.conversion.rpy2py(r_obj)

def _r_to_list(r_obj):
    return None if isinstance(r_obj, _sexp.NULLType) else list(r_obj)

    
_stats = _importr('stats')
_hclust_converts = {'merge': _r_to_numpy,
                    'height': _r_to_list,
                    'order': lambda x: _np.array(_r_to_list(x)),
                    'labels': lambda x: x,
                    'method': _r_to_list,
                    'call': _r_to_list,
                    'dist.method': _r_to_list}

def hclust(mat, method="euclidean"):
    rmat = _pandas_to_r(mat, as_matrix=True)
    hclust_obj = _stats.hclust(_stats.dist(rmat, method=method))
    hclust_obj = {k: _hclust_converts[k](v) for k, v in zip(hclust_obj.names, hclust_obj)}
    hclust_obj['labels'] = _np.array(mat.index.tolist())
    return hclust_obj


class DESeq2:
    
    deseq = _importr('DESeq2')
    bp = _importr('BiocParallel')
    to_dataframe = _robjects.r('function(x) data.frame(x)')
    
    threads = 1
    
    def __init__(self, count_data, meta_data, design_formula, threads=1):
        """
        DESeq2 wrapper written in rpy2
        
        :param count_data: Samples x Genes dataframe
        :type count_data: pd.DataFrame
        """
        self.gene_names = count_data.columns.copy()
        self.sample_names = count_data.index.copy()
        self.count_matrix, self.design_matrix, self.formula = self._deseq_to_r(count_data, meta_data, design_formula)
        self.threads = threads
        
    def run(self, **kwargs):
        self.de_obj = self._execute_deseq(self.count_matrix, self.design_matrix, self.formula, self.threads, **kwargs)
        return self
        
    def results(self, contrast, lfcThreshold=0, **kwargs):
        results = self._return_results(self.de_obj, contrast, lfcThreshold=lfcThreshold, **kwargs)
        results.index = self.gene_names
        return results
    
    def multiresults(self, contrast_func, contrast_iter, contrast_col, **kwargs):
        multires = []
        for x in contrast_iter:
            res = self.results(contrast_func(x), **kwargs)
            res[contrast_col] = x
            multires.append(res)
        return _pd.concat(multires)
    
    def results_names(self):
        return list(self.deseq.resultsNames(self.de_obj))

    @staticmethod
    def _deseq_to_r(count_data, meta_data, design_formula):
        
        count_matrix = _pandas_to_r(count_data, as_matrix=True, transpose=True)       
        design_matrix = _pandas_to_r(meta_data)
        formula = _Formula(design_formula)

        return count_matrix, design_matrix, formula

    @classmethod
    def _execute_deseq(cls, r_count, r_meta, r_design, threads=1, **kwargs):
        dds = cls.deseq.DESeqDataSetFromMatrix(countData=r_count, 
                                               colData=r_meta,
                                               design=r_design)
        return cls.deseq.DESeq(dds, BPPARAM=cls.bp.MulticoreParam(workers=threads), **kwargs)

    @classmethod
    def _return_results(cls, r_deseq, contrast, lfcThreshold=0.0, **kwargs):

        deseq_result = cls.deseq.results(r_deseq, contrast=_robjects.StrVector(contrast), lfcThreshold=float(lfcThreshold), **kwargs)
        return _r_to_pandas(cls.to_dataframe(deseq_result))

        
        