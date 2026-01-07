import numpy as np

class MatUtils:
    """
    Utility class for matrix operations, equivalent to MATLAB MatUtils class
    """
    
    @staticmethod
    def matrix_to_vec_idx_map(mat, mask):
        """
        Convert stack columns of a matrix as vector (skip elements not marked in the mask)
        
        Args:
            mat: Input matrix
            mask: Boolean mask matrix
            
        Returns:
            vec: Vector with masked elements
            idx_map: Index mapping
        """
        # Count the number of True elements in mask
        num_elements = np.sum(mask)
        
        # Initialize output arrays
        vec = np.full(num_elements, np.nan)
        idx_map = np.full((num_elements, 2), np.nan)
        
        idx = 0
        for j in range(mat.shape[1]):  # Columns
            for i in range(mat.shape[0]):  # Rows
                if mask[i, j] == 1:
                    vec[idx] = mat[i, j]
                    idx_map[idx, :] = [i, j]
                    idx += 1
                    
        return vec, idx_map
    
    @staticmethod
    def vec_idx_map_to_matrix(vec, idx_map, rows, cols, fill=np.nan):
        """
        Convert stacked vector to a matrix of rows x cols using the mapping of indexes idx_map
        
        Args:
            vec: Input vector
            idx_map: Index mapping
            rows: Number of rows in output matrix
            cols: Number of columns in output matrix
            fill: Fill value for unused elements
            
        Returns:
            matrix: Output matrix
        """
        # Initialize matrix with fill value
        matrix = np.full((rows, cols), fill)
        
        # Fill in values from vector according to index map
        for k in range(len(vec)):
            i, j = int(idx_map[k, 0]), int(idx_map[k, 1])
            matrix[i, j] = vec[k]
            
        return matrix