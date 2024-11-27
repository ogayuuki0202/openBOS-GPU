import numpy as np
from tqdm import tqdm
import torch      
from .reconstruction_utils import ART_torch  
from tqdm.contrib import tzip


def abel_transform_GPU(angle: np.ndarray, center: float, winy0: int, winy1: int, winx0: int, winx1: int, device: str):
    """
    Perform the Abel transform to convert refractive angle values into density differences.

    This function applies the Abel transform on a 2D array of refractive angles, adjusting the 
    values for background movement, calculating the distances from the center axis, and integrating 
    to derive density differences using the Gladstone-Dale constant. The calculation is done efficiently 
    on the GPU if available, using PyTorch tensors.

    Parameters
    ----------
    angle : np.ndarray
        A 2D numpy array representing refractive angles (radians) for each pixel in the image.
        
    center : float
        The y-axis index corresponding to the central axis of the transform, from which integration will proceed.
        
    winy0 : int
        The starting y-axis index of the region used to calculate the background mean.
        
    winy1 : int
        The ending y-axis index of the region used to calculate the background mean.
        
    winx0 : int
        The starting x-axis index of the region used to calculate the background mean.
        
    winx1 : int
        The ending x-axis index of the region used to calculate the background mean.
        
    device : str
        The device ('cpu' or 'cuda') on which the computation should be performed.

    Returns
    -------
    np.ndarray
        A 2D array of refractive index differences derived from the Abel transform.
        This represents the integrated density differences across the refractive angle image.
        
    Notes
    -----
    This function applies the Abel transform to a refractive angle image. The transformation involves:
    1. Subtracting the mean background value in the defined region (to compensate for background movement).
    2. Integrating the transformed angles along the radial distance, adjusted for axial symmetry.
    
    The function uses GPU acceleration if available for efficient computation on large images.

    Example
    -------
    abel_transform_GPU(angle_image, center=500, winy0=0, winy1=100, winx0=0, winx1=100, device='cuda')
    """
    
    # Step 1: Subtract the mean background value to compensate for background movement
    angle = angle - np.mean(angle[winy0:winy1, winx0:winx1])
    
    # Step 2: Remove values below the center, as they are not needed in the Abel transform
    angle = angle[0:center]
    
    # Step 3: Reverse the angle array so that the upper part corresponds to the central axis
    angle = angle[::-1]

    # Step 4: Convert the angle array to a PyTorch tensor on the specified device
    angle_tensor = torch.tensor(angle, device=device)

    # Step 5: Calculate the radial distance (η) for the integration, using PyTorch tensors
    eta_tensor = torch.arange(0, angle.shape[0] + 1, device=device)

    # Step 6: Create a tensor for r values (radial positions) up to the center
    r_tensor = torch.arange(0, center, device=device)

    # Step 7: Calculate the denominator A = √(η² - r²)
    A_tensor = eta_tensor**2 - r_tensor**2

    # Step 8: Compute the sliced tensor values of A (denominator of the Abel transform)
    A_r = _compute_sliced_tensors(A=A_tensor, device=device)
    
    # Step 9: Take the square root of A_r to obtain √(η² - r²)
    A_r = torch.sqrt(A_r)
    A_r = A_r.T  # Transpose for proper broadcasting in the next step
    
    # Step 10: Compute the integrand B = angle / (π * √(η² - r²)), where angle is the refractive angle tensor
    B_r = _compute_sliced_tensors_2D(angle_tensor,dim=0, device=device) / (A_r * np.pi)
    
    # Step 11: Perform the integration by summing over the radial axis (axis=0)
    ans = B_r.sum(axis=0)
    
    return ans



def ART_GPU(sinogram: np.ndarray, batch_size: int, device:str,reconstruction_angle : float, eps: float,tolerance:float =1e-24,max_stable_iters:int=1000000):
    """
    Perform Algebraic Reconstruction Technique (ART) on a sinogram using GPU.

    This function implements the ART algorithm for tomographic image reconstruction. 
    It iteratively refines the predicted reconstruction to minimize the difference 
    (residual) between the forward projection of the current prediction and the input sinogram.
    The process can utilize GPU acceleration for efficiency.

    Parameters:
        sinogram (np.ndarray): 
            Input sinogram with shape [N, Size, Angle], where:
            - N: Number of sinogram slices.
            - Size: Number of detector bins per projection.
            - Angle: Number of projections (angles).
            
        batch_size (int): 
            Number of slices processed in each batch. A batch size of 1 is recommended 
            if the CPU is used to avoid excessive memory usage.
            
        device (str): 
            Device for computation, either 'cuda' (for GPU) or 'cpu'.
            
        reconstruction_angle (float): 
            The angle spacing (in degrees) between consecutive projections in the sinogram.
            
        eps (float): 
            Convergence criterion for the iterative process. Iterations stop when the 
            maximum residual error across all pixels is below this value.
            
        tolerance (float): 
            Threshold for the change in residual error between iterations to consider 
            the convergence as stable. When the residual change remains below this 
            threshold for `max_stable_iters` iterations, the process is deemed stable.
            
        max_stable_iters (int): 
            Maximum number of iterations allowed with stable residuals (i.e., change in 
            residual error below the `tolerance` threshold) before stopping.

    Returns:
        torch.Tensor: 
            A reconstructed image tensor with shape [N, Image_Size, Image_Size], where 
            N corresponds to the number of input sinogram slices, and Image_Size is the 
            spatial resolution of the reconstructed image.
    """


    # Convert sinogram to a torch tensor and move it to the selected device
    sinogram_tensor = torch.FloatTensor(sinogram).permute(0, 2, 1).to(device)

    # Create data loaders for target and initial predictions
    target_dataloader = torch.utils.data.DataLoader(sinogram_tensor, batch_size=batch_size, shuffle=False)
    predict_dataloader = torch.utils.data.DataLoader(torch.zeros_like(sinogram_tensor), batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"target": target_dataloader, "predict": predict_dataloader}

    # Initialize the ART model with the input sinogram
    reconstruction_angle_radian = reconstruction_angle*np.pi/180
    model = ART_torch(sinogram=sinogram,reconstruction_angle=reconstruction_angle_radian)

    # Extract data loaders
    predict_dataloader = dataloaders_dict["predict"]
    target_dataloader = dataloaders_dict["target"]

    processed_batches = []

    # Convergence parameters

    prev_loss = float('inf')

    # Iterate through the data loader batches
    for i, (predict_batch, target_batch) in enumerate(tzip(predict_dataloader, target_dataloader)):
        # Move batches to the device
        predict_batch = predict_batch.to(model.device)
        target_batch = target_batch.to(model.device)
        stable_count = 0  # Counter for stable iterations

        iter_count = 0
        ATA = model.AT(model.A(torch.ones_like(predict_batch)))  # Precompute ATA for normalization
        ave_loss = torch.inf  # Initialize average loss

        # Initial loss calculation
        loss = torch.divide(model.AT(target_batch - model.A(predict_batch)), ATA)
        ave_loss = torch.max(torch.abs(loss)).item()

        # ART Iterative Reconstruction Loop
        while ave_loss > eps and stable_count < max_stable_iters:
            predict_batch = predict_batch + loss  # Update prediction
            ave_loss = torch.max(torch.abs(loss)).item()
            print("\r", f'Iteration: {iter_count}, Residual: {ave_loss}, Stable Count: {stable_count}', end="")
            iter_count += 1

            # Recalculate loss
            loss = torch.divide(model.AT(target_batch - model.A(predict_batch)), ATA)

            # Check residual change to update stable count
            if abs(ave_loss - prev_loss) < tolerance:
                stable_count += 1
            else:
                stable_count = 0

            prev_loss = ave_loss

        processed_batches.append(predict_batch)

    # Concatenate all processed batches along the batch dimension and return
    return torch.cat(processed_batches, dim=0)




def _compute_sliced_tensors(A, device=None):
    """
    Compute slices of the tensor A for each index r in parallel, such that each slice corresponds to A[r:a].
    This function is designed to run efficiently on both GPU and CPU.

    Args:
        A (torch.Tensor): Input 1D tensor from which slices are taken.
        device (torch.device, optional): Device on which to perform the computation.
                                         If None, the current device of A is used.

    Returns:
        list[torch.Tensor]: A list of tensors, where each tensor is the slice A[r:a] for r in range(len(A)).
    """
    # Use the device of A if no device is explicitly provided
    if device is None:
        device = A.device

    # Ensure the input tensor is on the correct device
    A = A.to(device)

    # Get the length of A
    a = A.size(0)

    # Create indices for slicing
    indices = torch.arange(a, device=device).unsqueeze(1)  # Column vector of r values
    arange_matrix = torch.arange(a, device=device).unsqueeze(0)  # Row vector [0, 1, ..., a-1]

    # Create a mask to determine the valid elements for each slice A[r:a]
    mask = arange_matrix >= indices  # True for elements to include

    # Apply the mask to extract elements in parallel and split into slices
    A_r = torch.masked_select(A.expand(a, a), mask).split(torch.arange(a, 0, -1, device=device).tolist())

    return A_r

def _compute_sliced_tensors_2D(A, dim=0, device=None):
    """
    Compute slices of the 2D tensor A along the specified dimension for each index r.
    Each slice corresponds to A[r:] or A[:,r:] depending on the dimension.

    Args:
        A (torch.Tensor): Input 2D tensor from which slices are taken.
        dim (int): Dimension along which to slice (0 for rows, 1 for columns).
        device (torch.device, optional): Device on which to perform the computation.
                                         If None, the current device of A is used.

    Returns:
        list[torch.Tensor]: A list of tensors, where each tensor is the slice A[r:] or A[:,r:].
    """
    # Ensure the input is on the correct device
    if device is None:
        device = A.device
    A = A.to(device)

    # Get the size of the slicing dimension
    size = A.size(dim)

    # Create indices for slicing
    indices = torch.arange(size, device=device).unsqueeze(1)  # Column vector of r values
    arange_matrix = torch.arange(size, device=device).unsqueeze(0)  # Row vector [0, 1, ..., size-1]

    # Create a mask to determine valid elements for each slice
    mask = arange_matrix >= indices  # True for elements to include

    # Expand A along the slicing dimension for parallel processing
    if dim == 0:  # Slice along rows
        expanded_A = A.unsqueeze(1).expand(size, size, -1)  # Add an extra dimension
        selected = torch.masked_select(expanded_A, mask.unsqueeze(-1))  # Apply mask
        A_r = selected.split(torch.arange(size, 0, -1, device=device).tolist())  # Split the slices
    elif dim == 1:  # Slice along columns
        expanded_A = A.unsqueeze(0).expand(size, -1, size)  # Add an extra dimension
        selected = torch.masked_select(expanded_A, mask.unsqueeze(0))  # Apply mask
        A_r = selected.split(torch.arange(size, 0, -1, device=device).tolist())  # Split the slices
    else:
        raise ValueError("dim must be 0 (rows) or 1 (columns).")

    return A_r