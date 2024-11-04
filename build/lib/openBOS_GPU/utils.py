from tqdm import trange
import torch


def compute_laplacian_chunk_2D_torch(array_chunk: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplacian of a given 2D chunk by calculating gradients 
    along specific axes and summing them.
    
    Parameters:
    array_chunk (Tensor): A chunk of the input tensor to compute the Laplacian on.

    Returns:
    Tensor: The Laplacian computed for the given chunk.
    """
    grad_yy = array_chunk[:, 2:] - 2 * array_chunk[:, 1:-1] + array_chunk[:, :-2]  # y-axis
    grad_zz = array_chunk[2:, :] - 2 * array_chunk[1:-1, :] + array_chunk[:-2, :]  # z-axis
    laplacian_chunk = grad_yy + grad_zz
    return laplacian_chunk

def compute_laplacian_chunk_3D_torch(array_chunk: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplacian of a given 3D chunk by calculating gradients 
    along specific axes and summing them.
    
    Parameters:
    array_chunk (Tensor): A chunk of the input tensor to compute the Laplacian on.

    Returns:
    Tensor: The Laplacian computed for the given chunk.
    """
    grad_xx = array_chunk[2:, :, :] - 2 * array_chunk[1:-1, :, :] + array_chunk[:-2, :, :]
    grad_yy = array_chunk[:, 2:, :] - 2 * array_chunk[:, 1:-1, :] + array_chunk[:, :-2, :]
    grad_zz = array_chunk[:, :, 2:] - 2 * array_chunk[:, :, 1:-1] + array_chunk[:, :, :-2]
    laplacian_chunk = grad_xx + grad_yy + grad_zz
    return laplacian_chunk

def compute_laplacian_in_chunks_2D_torch(array: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
    """
    Computes the Laplacian of an input 2D tensor in smaller chunks, allowing for 
    memory-efficient processing of large tensors.
    """
    shape = array.shape
    laplacian = torch.zeros_like(array)

    for i in trange(0, shape[0], chunk_size):            # Loop over x-axis in chunks
        for j in range(0, shape[1], chunk_size):         # Loop over y-axis in chunks
            chunk = array[i:i + chunk_size, j:j + chunk_size]
            laplacian_chunk = compute_laplacian_chunk_2D_torch(chunk)
            laplacian[i+1:i + chunk_size-1, j+1:j + chunk_size-1] = laplacian_chunk

    return laplacian

def compute_laplacian_in_chunks_3D_torch(array: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
    """
    Computes the Laplacian of an input 3D tensor in smaller chunks, allowing for 
    memory-efficient processing of large tensors.
    """
    shape = array.shape
    laplacian = torch.zeros_like(array)

    for i in trange(0, shape[0], chunk_size):            # Loop over x-axis in chunks
        for j in range(0, shape[1], chunk_size):         # Loop over y-axis in chunks
            for k in range(0, shape[2], chunk_size):     # Loop over z-axis in chunks
                chunk = array[i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                laplacian_chunk = compute_laplacian_chunk_3D_torch(chunk)
                laplacian[i+1:i + chunk_size-1, j+1:j + chunk_size-1, k+1:k + chunk_size-1] = laplacian_chunk

    return laplacian