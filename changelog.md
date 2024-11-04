# 11/04/2024

**Checkpoint Management:**

- Introduced `num_checkpoints` and `save_current_ckpt` parameters to control checkpoint saving behavior.
- Modified checkpoint saving logic to retain only the top `num_checkpoints` based on validation loss.
- Implemented `update_best_checkpoints` function to manage best checkpoints list, deleting older or less effective checkpoints.
- On resuming training, the script now scans the output directory to rebuild the `best_checkpoints` list, ensuring continuity in checkpoint management.
- Fixed variable initialization issues by moving `best_checkpoints` initialization outside the training loop and defining functions at the top level.
- Ensured that checkpoint management is handled only by the master process in DDP mode to prevent race conditions.

**Static Experts in MoE Model:**

- Added support for static experts in the MoE model by introducing the `static_experts` parameter.
- Modified the `MoE` class to include static experts in the forward pass, ensuring all tokens pass through them.
- Adjusted the gating mechanism to route tokens through both static and dynamic experts.
- Updated model configuration to include `static_experts`, allowing it to be set via the configuration file.
- Enhanced MoE usage logging to reflect the correct number of dynamic experts.

These changes improve the robustness of checkpoint management, preserving the best models based on validation loss across training sessions, and extend the MoE model's capabilities by integrating static experts.
