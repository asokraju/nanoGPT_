#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting script.sh execution..."

# Function to clean up background processes
cleanup() {
    echo "Stopping background processes..."
    # Kill all background jobs
    kill $(jobs -p) || true
    exit
}

# Trap termination signals to clean up background jobs
trap cleanup SIGINT SIGTERM

# Check if Python is installed
if ! command -v python &> /dev/null
then
    echo "Python not found. Installing Python..."
    apt-get update
    apt-get install -y python3 python3-pip
    ln -s /usr/bin/python3 /usr/bin/python
fi

# Upgrade pip
pip install --upgrade pip

# Install AWS CLI if not installed
if ! command -v aws &> /dev/null
then
    echo "AWS CLI not found. Installing AWS CLI..."
    pip install awscli
fi

# Install Python dependencies from requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing default packages..."
    pip install torch torchvision torchaudio numpy omegaconf wandb
fi

# Create the data directory if it doesn't exist
mkdir -p data

# Sync data from S3 into the data directory
echo "Syncing data from S3..."
aws s3 sync s3://your-data-s3-bucket/path/to/data ./data

# Verify data download
if [ "$(ls -A ./data)" ]; then
    echo "Data sync completed successfully."
else
    echo "Data sync failed or data directory is empty."
    exit 1
fi

# Start background sync process
echo "Starting background sync process to S3..."
(
    while true; do
        sleep 3600  # Sleep for 1 hour
        echo "Syncing 'out' directory to S3..."
        aws s3 sync ./out s3://your-output-s3-bucket/path/to/save
    done
) &  # Run in background

# Save PID of background process
SYNC_PID=$!

# Run your training script
echo "Running training script..."
python train_torch.py

# After training completes, kill the background sync process
echo "Training completed. Stopping background sync process..."
kill $SYNC_PID

echo "script.sh execution completed successfully."
