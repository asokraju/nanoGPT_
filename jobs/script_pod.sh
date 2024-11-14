#!/bin/bash

# Redirect all output to a log file while also displaying it in the console
exec > >(tee -i /home/eai_role/code/log.txt) 2>&1

# Exit immediately if a command exits with a non-zero status, and print each command before executing it
set -xe

# Signal that the container is starting up (used by Kubernetes readiness probe)
touch /tmp/ready

# Print current working directory
echo "Current directory: $PWD"

# List files to verify the correct files are present
echo "Listing files in local directory:"
ls -ltra

# Variables for S3 paths and local directories
S3_CODE_SRC="s3://your-s3-bucket/path/to/code/"
S3_DATA_SRC="s3://your-s3-bucket/path/to/data/"
S3_OUTPUT_DIR="s3://your-s3-bucket/path/to/output/"
LOCAL_CODE_DIR="/home/eai_role/code/"
LOCAL_DATA_DIR="/home/eai_role/code/data/"
LOCAL_OUTPUT_DIR="/home/eai_role/code/out/"
CHECKPOINT_DIR="/persistent_storage/checkpoints/"  # Directory for checkpoints

# Name and Python version for the conda environment
ENV_NAME="gpt-env"
PYTHON_VERSION="3.9"

# Sync code from S3 to local directory
echo "Syncing code from S3 to $LOCAL_CODE_DIR"
aws s3 sync "$S3_CODE_SRC" "$LOCAL_CODE_DIR"

# Sync data from S3 to local directory
echo "Syncing data from S3 to $LOCAL_DATA_DIR"
aws s3 sync "$S3_DATA_SRC" "$LOCAL_DATA_DIR"

# List files after syncing
echo "Listing files in $LOCAL_CODE_DIR:"
ls -R "$LOCAL_CODE_DIR"

# Initialize conda for use in this script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create a conda environment with the specified Python version
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate the conda environment
conda activate "$ENV_NAME"

# Install the required Python packages
pip install -r "$LOCAL_CODE_DIR/requirements.txt"

# NCCL environment variables for debugging (if using distributed training)
export NCCL_DEBUG=INFO

# Verify that the environment is activated and packages are installed
echo "Conda environment '$ENV_NAME' created and activated."
echo "Installed packages:"
pip list

# Download checkpoints from S3 if local checkpoint directory is empty
if [ -z "$(ls -A "$CHECKPOINT_DIR")" ]; then
  echo "No local checkpoints found. Syncing checkpoints from S3..."
  aws s3 sync "$S3_OUTPUT_DIR/checkpoints/" "$CHECKPOINT_DIR"
fi

# Function to sync checkpoints periodically to S3
sync_checkpoints() {
  while true; do
    sleep 600  # Sync every 10 minutes
    echo "Periodic checkpoint sync to S3..."
    aws s3 sync "$CHECKPOINT_DIR" "$S3_OUTPUT_DIR/checkpoints/"
  done
}

# Start the background process to sync checkpoints
sync_checkpoints &

# Run the training script using torchrun for distributed training
echo "Running training script with torchrun..."
torchrun --standalone --nproc_per_node=4 "$LOCAL_CODE_DIR/train.py" --checkpoint_dir "$CHECKPOINT_DIR"

# Kill the background checkpoint sync process after training completes
pkill -f sync_checkpoints

# Signal that the application is healthy (used by Kubernetes liveness probe)
touch /tmp/healthy

# Final sync to ensure all outputs and checkpoints are copied after training completes
echo "Syncing output data to S3..."
aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_DIR/out/"
echo "Data is saved to S3 at $S3_OUTPUT_DIR/out/"

echo "Syncing checkpoints to S3..."
aws s3 sync "$CHECKPOINT_DIR" "$S3_OUTPUT_DIR/checkpoints/"
echo "Checkpoints are saved to S3 at $S3_OUTPUT_DIR/checkpoints/"

# Save the log file to S3
aws s3 cp "$LOCAL_CODE_DIR/log.txt" "$S3_OUTPUT_DIR/log.txt"
echo "Log file is saved to S3 at $S3_OUTPUT_DIR/log.txt"
