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

# Create the output directory if it doesn't exist
mkdir -p "$LOCAL_OUTPUT_DIR"

# Download checkpoints from S3 to out directory if available
echo "Syncing checkpoints from S3 to local out directory..."
aws s3 sync "$S3_OUTPUT_DIR" "$LOCAL_OUTPUT_DIR"

# Function to sync outputs and checkpoints periodically to S3
sync_outputs() {
  while true; do
    sleep 600  # Sync every 10 minutes
    echo "Periodic output sync to S3..."
    aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_DIR"
  done
}

# Start the background process to sync outputs
sync_outputs &

# Run the training script using torchrun for distributed training
echo "Running training script with torchrun..."
torchrun --standalone --nproc_per_node=4 "$LOCAL_CODE_DIR/train.py" --output_dir "$LOCAL_OUTPUT_DIR"

# Kill the background output sync process after training completes
pkill -f sync_outputs

# Signal that the application is healthy (used by Kubernetes liveness probe)
touch /tmp/healthy

# Final sync to ensure all outputs and checkpoints are copied after training completes
echo "Syncing output data to S3..."
aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_DIR"
echo "Data is saved to S3 at $S3_OUTPUT_DIR"

# Save the log file to S3
aws s3 cp "$LOCAL_CODE_DIR/log.txt" "$S3_OUTPUT_DIR/log.txt"
echo "Log file is saved to S3 at $S3_OUTPUT_DIR/log.txt"
