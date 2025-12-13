"""
QG - Python wrapper for RaBitQ QuantizedGraph
"""
import numpy as np
from typing import Optional, Union

try:
    from ._qg_core import QuantizedGraphCore
except ImportError:
    raise ImportError(
        "Failed to import _qg_core. Please ensure the package is properly built. "
        "Run 'pip install -e .' or 'python setup.py build_ext --inplace'"
    )


class QG:
    """
    QuantizedGraph Python interface.
    
    This class provides a Python interface to the RaBitQ QuantizedGraph
    for approximate nearest neighbor search.
    
    Example:
        >>> from qg import QG
        >>> qg = QG()
        >>> qg.load_index("output/qg_metadata.bin", "output/qg_data.bin")
        >>> query = np.random.rand(128).astype(np.float32)
        >>> results = qg.search_one(query, k=10, L=100)
        >>> print(results)
    """
    
    def __init__(self):
        """Initialize an empty QG instance."""
        self._core = QuantizedGraphCore()
        self._ef = 0
    
    def load_index(self, meta_filename: str, data_filename: str):
        """
        Load index from metadata and data files.
        
        Args:
            meta_filename: Path to the metadata file (e.g., qg_metadata.bin)
            data_filename: Path to the data file (e.g., qg_data.bin)
            
        Raises:
            RuntimeError: If the index files cannot be loaded
        """
        self._core.load_index(meta_filename, data_filename)
    
    def set_ef(self, ef: int):
        """
        Set the ef (exploration factor) parameter for search.
        
        The ef parameter controls the tradeoff between search quality and speed.
        Higher values give better recall but slower search.
        
        Args:
            ef: Exploration factor (typically 40-200)
        """
        self._ef = ef
        self._core.set_ef(ef)
    
    def search_one(self, query: np.ndarray, k: int, L: int) -> np.ndarray:
        """
        Search for k nearest neighbors for a single query vector.
        
        This corresponds to the search_one function in load_test_qg.cpp.
        
        Args:
            query: Query vector as numpy array (1D, float32)
            k: Number of nearest neighbors to return
            L: Exploration factor (ef parameter) for this search
            
        Returns:
            Array of k nearest neighbor IDs (uint32)
            
        Example:
            >>> query = np.random.rand(128).astype(np.float32)
            >>> results = qg.search_one(query, k=10, L=100)
            >>> print(f"Found {len(results)} neighbors: {results}")
        """
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim != 1:
            raise ValueError("Query must be 1-dimensional")
        
        # Set ef parameter for this search
        self._core.set_ef(L)
        
        return self._core.search(query, k)
    
    def search_all(self, queries: np.ndarray, k: int, L: int) -> np.ndarray:
        """
        Search for k nearest neighbors for multiple query vectors.
        
        This corresponds to the search_all function in load_test_qg.cpp.
        
        Args:
            queries: Query vectors as numpy array (2D, shape [n_queries, dim], float32)
            k: Number of nearest neighbors to return per query
            L: Exploration factor (ef parameter) for this search
            
        Returns:
            Array of shape [n_queries, k] containing nearest neighbor IDs (uint32)
            
        Example:
            >>> queries = np.random.rand(100, 128).astype(np.float32)
            >>> results = qg.search_all(queries, k=10, L=100)
            >>> print(f"Shape: {results.shape}")  # (100, 10)
        """
        if not isinstance(queries, np.ndarray):
            queries = np.array(queries, dtype=np.float32)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if queries.ndim != 2:
            raise ValueError("Queries must be 2-dimensional [n_queries, dim]")
        
        # Set ef parameter for this search
        self._core.set_ef(L)
        
        n_queries = queries.shape[0]
        results = np.zeros((n_queries, k), dtype=np.uint32)
        
        for i in range(n_queries):
            results[i] = self._core.search(queries[i], k)
        
        return results
    
    @property
    def num_vertices(self) -> int:
        """Get the number of vertices in the graph."""
        return self._core.num_vertices()
    
    @property
    def dimension(self) -> int:
        """Get the dimension of vectors."""
        return self._core.dimension()
    
    @property
    def degree_bound(self) -> int:
        """Get the degree bound of the graph."""
        return self._core.degree_bound()
    
    @property
    def entry_point(self) -> int:
        """Get the entry point of the graph."""
        return self._core.entry_point()
    
    @property
    def ef(self) -> int:
        """Get the current ef parameter."""
        return self._ef


__all__ = ['QG']
