# --- START OF FILE ep_cfg_script.py ---
import logging
import sys
import traceback
import gradio as gr
from modules import scripts, script_callbacks
from functools import partial
from typing import Any

# Import the patching class from the CFGZERO directory within the extension
try:
    from EPCFG.nodes_ep_cfg import EP_CFG_Node
except ImportError:
    # Fallback for local testing or different setups where direct import might fail
    # This ensures the script can find the node definition if the extension's
    # root isn't automatically added to sys.path by WebUI.
    import os
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    sys.path.append(parent_dir)
    from EPCFG.nodes_ep_cfg import EP_CFG_Node
    logging.warning("EP-CFG: Added parent directory to sys.path for module import.")


class EP_CFG_Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.zero_init_first_step = False
        self.l_percentile = 0.45 # Default from paper
        self.h_percentile = 0.55 # Default from paper
        self.ep_cfg_node_instance = EP_CFG_Node() # Instantiate the node logic

    # Lower sorting priority means it runs earlier.
    # This is important for model patches to be applied before samplers use the model.
    sorting_priority = 15.2 

    def title(self):
        return "EP-CFG Guidance"

    def show(self, is_img2img):
        # Always visible in both txt2img and img2img tabs
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        """
        Creates the Gradio UI for the extension.
        """
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Toggle EP-CFG (Energy-Preserving CFG) guidance. Modifies how conditional and unconditional guidance are combined to prevent over-saturation. Based on <a href='https://arxiv.org/abs/2412.09966v1' target='_blank'>arXiv:2412.09966v1</a>.</i></p>")
            
            enabled = gr.Checkbox(label="Enable EP-CFG", value=self.enabled)
            zero_init_first_step = gr.Checkbox(label="Zero Init First Step (Experimental)", value=self.zero_init_first_step,
                                               info="Sets the first step's noise prediction to zero for unconditional guidance. Can help with initial image structure.")
            
            with gr.Row():
                l_percentile = gr.Slider(label="Lower Percentile (l)", minimum=0.0, maximum=1.0, step=0.01, value=self.l_percentile,
                                         info="Lower percentile for robust energy estimation. Values outside [l, h] are ignored. Paper recommends 0.45.")
                h_percentile = gr.Slider(label="Upper Percentile (h)", minimum=0.0, maximum=1.0, step=0.01, value=self.h_percentile,
                                         info="Upper percentile for robust energy estimation. Values outside [l, h] are ignored. Paper recommends 0.55.")

        # Attach change listeners to update internal script state
        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled],
            outputs=None
        )
        zero_init_first_step.change(
            lambda x: self.update_zero_init(x),
            inputs=[zero_init_first_step],
            outputs=None
        )
        l_percentile.change(
            lambda x: self.update_l_percentile(x),
            inputs=[l_percentile],
            outputs=None
        )
        h_percentile.change(
            lambda x: self.update_h_percentile(x),
            inputs=[h_percentile],
            outputs=None
        )

        # Store controls for process_before_every_sampling to retrieve their values
        self.ui_controls = [enabled, zero_init_first_step, l_percentile, h_percentile]
        return self.ui_controls

    # UI update methods
    def update_enabled(self, value):
        self.enabled = value
        logging.debug(f"EP-CFG: Enabled toggled to: {self.enabled}")

    def update_zero_init(self, value):
        self.zero_init_first_step = value
        logging.debug(f"EP-CFG: Zero Init First Step toggled to: {self.zero_init_first_step}")

    def update_l_percentile(self, value):
        self.l_percentile = value
        logging.debug(f"EP-CFG: Lower Percentile toggled to: {self.l_percentile}")

    def update_h_percentile(self, value):
        self.h_percentile = value
        logging.debug(f"EP-CFG: Upper Percentile toggled to: {self.h_percentile}")

    def process_before_every_sampling(self, p, *args, **kwargs):
        """
        This method is called before each sampling process (e.g., before generating an image).
        It applies or removes the EP-CFG patch based on the UI settings.
        """
        # Retrieve values from UI controls (passed as *args)
        if len(args) >= 4:
            self.enabled = args[0]
            self.zero_init_first_step = args[1]
            self.l_percentile = args[2]
            self.h_percentile = args[3]
        else:
            logging.warning("EP-CFG: Not enough arguments provided to process_before_every_sampling, using current UI values.")

        # Handle XYZ Grid integration: override UI values if XYZ grid is active
        xyz_settings = getattr(p, "_ep_cfg_xyz", {}) # Internal attribute to store XYZ grid values
        if "enabled" in xyz_settings:
            self.enabled = xyz_settings["enabled"].lower() == "true"
        if "zero_init" in xyz_settings:
            self.zero_init_first_step = xyz_settings["zero_init"].lower() == "true"
        if "l_percentile" in xyz_settings:
            try:
                self.l_percentile = float(xyz_settings["l_percentile"])
            except ValueError:
                logging.warning(f"EP-CFG: Invalid l_percentile value from XYZ Grid: {xyz_settings['l_percentile']}")
        if "h_percentile" in xyz_settings:
            try:
                self.h_percentile = float(xyz_settings["h_percentile"])
            except ValueError:
                logging.warning(f"EP-CFG: Invalid h_percentile value from XYZ Grid: {xyz_settings['h_percentile']}")

        # Crucial for proper patching/unpatching:
        # Always start with a fresh clone of the original UNet if we've previously modified it.
        # This ensures that if EP-CFG is disabled, or if other scripts modify the UNet,
        # we are starting from a known, clean state for our patch.
        if hasattr(p, '_original_unet_before_ep_cfg'):
            p.sd_model.forge_objects.unet = p._original_unet_before_ep_cfg.clone()
        else:
            # Store the state of the UNet *before* our script potentially modifies it for the first time
            p._original_unet_before_ep_cfg = p.sd_model.forge_objects.unet.clone()
        
        unet_to_patch = p.sd_model.forge_objects.unet # This is now a fresh clone or the original UNet

        # If EP-CFG is disabled, remove the patch if it was previously applied
        if not self.enabled:
            if hasattr(unet_to_patch, "_ep_cfg_patched"): # Check for our custom marker
                # Remove the specific post_cfg_function we added
                unet_to_patch.set_model_sampler_post_cfg_function(None, "ep_cfg_guidance")
                delattr(unet_to_patch, "_ep_cfg_patched") # Remove our marker
            p.sd_model.forge_objects.unet = unet_to_patch # Ensure the clean UNet is set
            # Clean up extra_generation_params to avoid clutter in PNG info
            for key in ["ep_cfg_enabled", "ep_cfg_init_first_step", "ep_cfg_l_percentile", "ep_cfg_h_percentile"]:
                if key in p.extra_generation_params:
                    del p.extra_generation_params[key]
            logging.debug(f"EP-CFG: Disabled. UNet restored.")
            return

        # If EP-CFG is enabled, apply the patch
        logging.debug(f"EP-CFG: Enabling with Zero Init: {self.zero_init_first_step}, L: {self.l_percentile}, H: {self.h_percentile}")
        
        # Call the patch method of our EP_CFG_Node instance
        # It returns a tuple (model,), so we take the first element [0]
        patched_unet = self.ep_cfg_node_instance.patch(
            unet_to_patch, # Pass the current UNet (which should be a fresh clone)
            zero_init_first_step=self.zero_init_first_step,
            l_percentile=self.l_percentile,
            h_percentile=self.h_percentile
        )[0]
        
        p.sd_model.forge_objects.unet = patched_unet # Assign the patched UNet back to the sampler
        setattr(p.sd_model.forge_objects.unet, "_ep_cfg_patched", True) # Mark the UNet as patched by us

        # Add parameters to generation info (PNG info)
        p.extra_generation_params.update({
            "ep_cfg_enabled": True,
            "ep_cfg_init_first_step": self.zero_init_first_step,
            "ep_cfg_l_percentile": self.l_percentile,
            "ep_cfg_h_percentile": self.h_percentile,
        })
        logging.debug(f"EP-CFG: Enabled. UNet Patched.")
        return

# --- XYZ Grid Integration ---
# This section adds options for EP-CFG to the XYZ Plot script
def ep_cfg_set_value(p, x: Any, xs: Any, *, field: str):
    """Helper function to set values for XYZ grid."""
    if not hasattr(p, "_ep_cfg_xyz"):
        p._ep_cfg_xyz = {}
    # XYZ grid typically sends values as strings, store them as such.
    # The process_before_every_sampling method will parse them back.
    p._ep_cfg_xyz[field] = str(x)

def make_ep_cfg_axis_on_xyz_grid():
    """Registers EP-CFG options with the XYZ Grid script."""
    xyz_grid = None
    # Find the XYZ Grid script module
    for script_data in scripts.scripts_data:
        if script_data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py") : # Support both common names
            xyz_grid = script_data.module
            break

    if xyz_grid is None:
        logging.warning("EP-CFG: XYZ Grid script not found. XYZ options will not be available.")
        return

    # Prevent duplicate registration if called multiple times
    if any(x.label.startswith("(EP-CFG)") for x in xyz_grid.axis_options):
        logging.info("EP-CFG: XYZ Grid options already registered.")
        return
        
    # Define the XYZ Axis Options for EP-CFG
    ep_cfg_options = [
        xyz_grid.AxisOption(
            label="(EP-CFG) Enabled",
            type=str, # XYZ grid expects string for boolean choices
            apply=partial(ep_cfg_set_value, field="enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            label="(EP-CFG) Zero Init First Step",
            type=str,
            apply=partial(ep_cfg_set_value, field="zero_init"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            label="(EP-CFG) L Percentile",
            type=float, # Use float type for XYZ grid to allow direct float input
            apply=partial(ep_cfg_set_value, field="l_percentile"),
            choices=lambda: [0.45, 0.40, 0.50, 0.35, 0.60] # Example choices
        ),
        xyz_grid.AxisOption(
            label="(EP-CFG) H Percentile",
            type=float,
            apply=partial(ep_cfg_set_value, field="h_percentile"),
            choices=lambda: [0.55, 0.50, 0.60, 0.45, 0.65] # Example choices
        ),
    ]
    xyz_grid.axis_options.extend(ep_cfg_options)
    logging.info("EP-CFG: XYZ Grid options successfully registered.")


def on_ep_cfg_before_ui():
    """Callback executed before the WebUI UI is built."""
    try:
        make_ep_cfg_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] EP-CFG Script: Error setting up XYZ Grid options:\n{error}",
            file=sys.stderr,
        )

# Register the callback to run before the UI is built
script_callbacks.on_before_ui(on_ep_cfg_before_ui)
# --- END OF FILE ep_cfg_script.py ---
