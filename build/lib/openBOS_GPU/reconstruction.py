import numpy as np
from tqdm import tqdm
import torch      
from .reconstruction_utils import ART_torch  
from tqdm.contrib import tzip                              

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
