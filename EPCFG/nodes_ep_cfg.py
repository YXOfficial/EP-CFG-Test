# --- START OF FILE nodes_ep_cfg.py (CORRECTED) ---
import torch
import logging

class EP_CFG_Node:
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the EP-CFG node.
        - model: The UNet model to be patched.
        - zero_init_first_step: Boolean to enable/disable zero initialization for the first step.
        - l_percentile: Lower percentile for robust energy estimation (from paper).
        - h_percentile: Upper percentile for robust energy estimation (from paper).
        """
        return {
            "required": {
                "model": ("MODEL",),
                "zero_init_first_step": ("BOOLEAN", {"default": False}),
                "l_percentile": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "h_percentile": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies EP-CFG (Energy-Preserving CFG) guidance scaling based on arXiv:2412.09966v1."

    def calculate_robust_energy(self, tensor: torch.Tensor, l_percentile: float, h_percentile: float) -> torch.Tensor:
        """
        Calculates the robust energy for a batch of latents as defined in the EP-CFG paper.
        Erobust = sum(x_i^2 * 1[P_l <= x_i <= P_h])
        Only values within the specified percentile range contribute to the energy sum.

        Args:
            tensor (torch.Tensor): The input tensor (e.g., denoised latent) of shape (B, C, H, W).
            l_percentile (float): The lower percentile (e.g., 0.45).
            h_percentile (float): The upper percentile (e.g., 0.55).

        Returns:
            torch.Tensor: A tensor of shape (B, 1) containing the robust energy for each item in the batch.
        """
        batch_size = tensor.shape[0]
        # Reshape to (batch_size, -1) for percentile calculation across each latent
        tensor_flat = tensor.reshape(batch_size, -1)

        # Calculate percentiles for each latent in the batch
        # keepdim=True ensures the output shape is (batch_size, 1) for broadcasting
        lower_bounds = torch.quantile(tensor_flat, l_percentile, dim=1, keepdim=True)
        upper_bounds = torch.quantile(tensor_flat, h_percentile, dim=1, keepdim=True)

        # Create a mask for values within the percentile range for each latent
        mask = (tensor_flat >= lower_bounds) & (tensor_flat <= upper_bounds)

        # Apply mask, square the contributing values, and sum them up.
        # This correctly implements sum(x_i^2 * 1[...])
        robust_squared_values = (tensor_flat ** 2) * mask.float()
        energies = torch.sum(robust_squared_values, dim=1, keepdim=True) # Sum over flattened dimensions

        return energies

    def patch(self, model, zero_init_first_step: bool = False, l_percentile: float = 0.45, h_percentile: float = 0.55):
        """
        Patches the model to apply EP-CFG guidance.

        Args:
            model: The model object (e.g., UNet model wrapper from Forge).
            zero_init_first_step (bool): If True, applies zero initialization for the first sampling step.
            l_percentile (float): Lower percentile for robust energy calculation.
            h_percentile (float): Upper percentile for robust energy calculation.

        Returns:
            tuple: A tuple containing the patched model.
        """
        m = model.clone() # Clone the model to avoid modifying the original in place

        # Capture initial sigma for first step detection (used for zero_init_first_step)
        try:
            initial_sigma = m.model.model_sampling.sigma_max 
        except AttributeError:
            logging.warning("EP-CFG: Could not determine initial_sigma from model. First step zero-init might not work correctly.")
            initial_sigma = float('inf') # Set to infinity to make first step detection unlikely if sigma_max is not found

        # This is the function that will be called by the sampler after it computes
        # the conditional and unconditional noise predictions (or denoised latents).
        def ep_cfg_guidance_function(args):
            cond_scale = args['cond_scale']
            cond_denoised = args['cond_denoised']     # Corresponds to xc in the paper
            uncond_denoised = args['uncond_denoised'] # Corresponds to xu in the paper
            # args also contains 'sigma' (current noise level) and 'x' (current latent)

            # Make a mutable copy of uncond_denoised for potential modification
            _uncond_denoised_for_cfg = uncond_denoised
            
            # --- Zero Init First Step Logic ---
            # This modifies the unconditional prediction for the first step.
            if zero_init_first_step:
                current_sigma_val = args['sigma'][0].item() # Get scalar value of current sigma
                # Compare with a small tolerance due to potential float precision issues
                if abs(current_sigma_val - initial_sigma) < 1e-5:
                    logging.debug(f"EP-CFG: Applying zero_init for first step (sigma: {current_sigma_val})")
                    # Set the unconditional prediction to zeros for this step's CFG calculation
                    _uncond_denoised_for_cfg = torch.zeros_like(uncond_denoised)

            # Ensure original_shape and batch_size are correctly derived
            original_shape = cond_denoised.shape
            if len(original_shape) < 2: # Expected shape is at least (B, C, ...)
                logging.error("EP-CFG: Unexpected tensor shape for denoised latents. Falling back to standard CFG.")
                # Fallback to standard CFG if the input shape is problematic
                return _uncond_denoised_for_cfg + cond_scale * (cond_denoised - _uncond_denoised_for_cfg)

            batch_size = original_shape[0]

            # --- Step 1: Calculate Xcfg_original (standard CFG output) ---
            # Use the potentially modified _uncond_denoised_for_cfg here
            Xcfg_original = _uncond_denoised_for_cfg + cond_scale * (cond_denoised - _uncond_denoised_for_cfg)

            # --- Step 2: Calculate Ec = ||xc||^2 (robustly) ---
            Ec = self.calculate_robust_energy(cond_denoised, l_percentile, h_percentile)
            
            # --- Step 3: Calculate Ecfg = ||Xcfg_original||^2 (robustly) ---
            Ecfg = self.calculate_robust_energy(Xcfg_original, l_percentile, h_percentile)

            # --- Step 4: Calculate scaling factor = sqrt(Ec / Ecfg) ---
            # Add a small epsilon (1e-8) for numerical stability to prevent division by zero
            scale_factor = torch.sqrt(Ec / (Ecfg + 1e-8))
            
            # Reshape scale_factor for broadcasting with the original tensor shapes (B, C, H, W)
            # (batch_size, 1) -> (batch_size, 1, 1, 1) for 4D tensors
            scale_factor_reshaped = scale_factor.view(batch_size, *([1] * (len(original_shape) - 1)))

            # --- Step 5: Apply scaling: Xcfg_rescaled = Xcfg_original * scale_factor_reshaped ---
            Xcfg_rescaled = Xcfg_original * scale_factor_reshaped
            
            return Xcfg_rescaled

        # Apply the custom guidance function to the model.
        # "ep_cfg_guidance" is a unique identifier for this specific patch.
        m.set_model_sampler_post_cfg_function(ep_cfg_guidance_function, "ep_cfg_guidance")
        logging.debug(f"EP-CFG: Model patched with zero_init_first_step = {zero_init_first_step}, l_percentile = {l_percentile}, h_percentile = {h_percentile}, initial_sigma_captured = {initial_sigma}")
        return (m,) # Return the patched model as a tuple

# For ComfyUI compatibility (optional, but good practice for potential future use)
NODE_CLASS_MAPPINGS = {
    "EP_CFG_Node": EP_CFG_Node
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EP_CFG_Node": "YX-EP-CFG Guidance Patcher"
}

# --- END OF FILE nodes_ep_cfg.py (CORRECTED) ---
