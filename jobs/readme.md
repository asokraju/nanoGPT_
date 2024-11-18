# Report: Deploying a GPU-Accelerated Training Job on Kubernetes

## Introduction

This report provides a comprehensive, step-by-step guide to deploying a GPU-accelerated machine learning training job on Kubernetes. It aims to bridge the gap between understanding individual components and grasping how they integrate to form a cohesive, scalable, and efficient training environment. Specifically, it covers the following:

- **An Integrated Overview of the Deployment Workflow**
- **Understanding the Kubernetes Job and Its Components**
- **Explaining the `script.sh` File Used for Training**
- **Detailed Walkthrough of the Kubernetes YAML Configuration**
- **Instructions on How to Deploy the Job to a Kubernetes Cluster**

By the end of this report, we should have a clear understanding of how to set up and run a GPU-intensive training task on Kubernetes using the provided configurations, and how each component interacts within the system to achieve this goal.

---

## Table of Contents

1. [Integrated Overview of the Deployment Workflow](#1-integrated-overview-of-the-deployment-workflow)
2. [Overview of Kubernetes Concepts](#2-overview-of-kubernetes-concepts)
3. [File Layout and Directory Structure](#3-file-layout-and-directory-structure)
   - 3.1 [Files in Amazon S3](#31-files-in-amazon-s3)
   - 3.2 [Directory Structure in the Kubernetes Pod](#32-directory-structure-in-the-kubernetes-pod)
4. [The Training Script (`script.sh`)](#4-the-training-script-scriptsh)
   - 4.1 [Explanation of `script.sh`](#41-explanation-of-scriptsh)
5. [Kubernetes Job Configuration (`job.yaml`)](#5-kubernetes-job-configuration-jobyaml)
   - 5.1 [Explanation of `job.yaml`](#51-explanation-of-jobyaml)
6. [Deployment Steps](#6-deployment-steps)
7. [Conclusion](#7-conclusion)

---

## 1. Integrated Overview of the Deployment Workflow

Deploying a GPU-accelerated training job on Kubernetes involves several interconnected components working together to orchestrate, execute, and manage the training process. This section provides a holistic view of how these components interact, laying the foundation for the detailed explanations that follow.

### Workflow Summary

1. **Code and Data Storage in Amazon S3**: The training code, data, and configuration files are stored in Amazon S3 buckets. This centralized storage ensures scalability and accessibility for the training job.

2. **Kubernetes Job Definition (`job.yaml`)**: The Kubernetes Job is defined in a YAML configuration file, specifying the resources required, container image, and commands to execute.

3. **Script Execution (`script.sh`)**: A shell script orchestrates the setup within the container, including environment setup, data synchronization, and initiation of the training process.

4. **ConfigMap Creation**: The `script.sh` file is injected into the Kubernetes Pod using a ConfigMap, allowing the script to be executed inside the container.

5. **Pod Creation and Execution**: When the Job is deployed, Kubernetes schedules a Pod based on the specifications in `job.yaml`. The Pod runs the container, mounts volumes, and executes `script.sh`.

6. **Environment Setup Inside the Pod**:
   - **Data Synchronization**: The script downloads the code and data from S3 into the Pod's filesystem.
   - **Environment Configuration**: A Conda environment is created and activated, and dependencies are installed.
   - **Checkpoint Handling**: If previous checkpoints exist, they are downloaded from S3 to resume training.

7. **Training Execution**: The training script (`train.py`) is executed using `torchrun`, leveraging multiple GPUs as specified.

8. **Output and Checkpoint Management**:
   - **Periodic Syncing**: Outputs and checkpoints are periodically synced back to S3 during training to prevent data loss.
   - **Final Syncing**: Upon completion, all outputs and logs are synced back to S3 for persistence.

9. **Health Monitoring**:
   - **Probes**: Kubernetes uses readiness and liveness probes to monitor the Pod's status based on files created by `script.sh`.
   - **Automatic Recovery**: If the Pod fails, Kubernetes can restart it based on the `backoffLimit` and restart policies.

10. **Resource Management**:
    - **GPU Allocation**: The Job requests specific GPU resources, ensuring the Pod is scheduled on nodes with the required hardware.
    - **Resource Limits**: CPU and memory requests and limits are set to optimize scheduling and prevent resource contention.

### Component Interaction

- **Kubernetes and Docker**: Kubernetes uses the specified Docker image to create a containerized environment for the training job.

- **ConfigMap and `script.sh`**: The ConfigMap injects `script.sh` into the container, allowing the script to control the setup and execution flow within the Pod.

- **Amazon S3 and Data Management**: S3 serves as the centralized storage for code, data, and outputs, enabling the Pod to pull necessary files and push outputs regardless of where it is scheduled.

- **Conda Environment and Dependencies**: The script creates a Conda environment inside the container, ensuring all Python dependencies are met without modifying the container image.

- **Training Script and GPUs**: The `train.py` script utilizes GPUs allocated by Kubernetes to perform intensive computations, facilitated by `torchrun` for distributed training.

- **Health Probes and Kubernetes**: The creation of `/tmp/ready` and `/tmp/healthy` by `script.sh` allows Kubernetes to monitor the Pod's readiness and liveness, ensuring the application is functioning correctly.

### The Big Picture

By combining these components, we achieve a scalable, resilient, and efficient system for training machine learning models:

- **Scalability**: Kubernetes orchestrates resources across the cluster, allowing for horizontal scaling as needed.

- **Resilience**: Health probes and restart policies enable automatic recovery from failures, ensuring long-running training jobs can continue despite transient issues.

- **Efficiency**: GPU allocation and resource limits ensure optimal utilization of hardware resources, while periodic syncing prevents data loss without excessive overhead.

This integrated approach leverages the strengths of Kubernetes, containerization, and cloud storage to facilitate complex training workflows in a manageable and reproducible manner.

---

## 2. Overview of Kubernetes Concepts

Before diving into the configurations, let's briefly cover some Kubernetes concepts relevant to this deployment.

### Kubernetes Components Used:

- **Pod**: The smallest deployable unit in Kubernetes, which can contain one or more containers.

- **Job**: A controller that creates one or more Pods to run a finite task to completion.

- **Container**: An instance of a Docker image running within a Pod.

- **ConfigMap**: A Kubernetes object to store non-confidential configuration data in key-value pairs.

- **Volume**: A directory, possibly with data in it, accessible to the containers in a Pod.

- **Resource Requests and Limits**: Specifications to inform Kubernetes about the minimum and maximum resources (CPU, memory, GPU) a container needs.

- **Probes (Liveness and Readiness)**: Mechanisms to check the health and readiness of a container.

---

## 3. File Layout and Directory Structure

Understanding the organization of files and directories is crucial for managing the code, data, and outputs of your training job. This section details the file layout both in Amazon S3 and within the Kubernetes Pod after copying the files. It also explains how outputs such as logs and checkpoints are stored.

### 3.1 Files in Amazon S3

#### **Code Directory Structure**

Your main code folder stored in Amazon S3 contains the following files:

```
s3://your-s3-bucket/path/to/code/
├── model.py
├── train.py
├── requirements.txt
├── config/
│   └── conf.yaml
└── job.yaml
```

- **model.py**: Contains the model architecture definition.
- **train.py**: The main script for training the model.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **config/conf.yaml**: Configuration file with training parameters and settings.
- **job.yaml**: Kubernetes Job configuration file (used locally, not in the Pod).

#### **Data Directory Structure**

Your data folder in Amazon S3 contains the training and validation datasets:

```
s3://your-s3-bucket/path/to/data/
├── train.bin
└── val.bin
```

- **train.bin**: Serialized training data.
- **val.bin**: Serialized validation data.

### 3.2 Directory Structure in the Kubernetes Pod

After copying the files from Amazon S3 to the Pod, the directory structure within the container is organized as follows:

#### **Local Code Directory**

```
/home/eai_role/code/
├── model.py
├── train.py
├── requirements.txt
├── config/
│   └── conf.yaml
├── data/
│   ├── train.bin
│   └── val.bin
├── out/  # Generated during execution
│   ├── log.txt
│   ├── current_ckpt.pt
│   ├── checkpoint_1.pt
│   ├── checkpoint_2.pt
│   └── events.out.tfevents.xxxxx
├── script.sh
└── job.yaml  # Not required here, copied for completeness
```

- **/home/eai_role/code/**: The directory where code files are stored.
- **/home/eai_role/code/data/**: The directory where data files are stored.
- **/home/eai_role/code/out/**: The main output directory.
  - **checkpoints/**: Stores model checkpoints saved during training.
    - **checkpoint_*.pt**: Checkpoint files (PyTorch model state dictionaries).
  - **logs/**: Contains training logs and other log files.
    - **training_log.txt**: Detailed log of the training process.
  - **tensorboard/**: Stores TensorBoard log files for visualization.
    - **events.out.tfevents.xxxxx**: Event files used by TensorBoard.

---

By understanding the file layout and directory structure, we can better navigate the environment within the Kubernetes Pod, making it easier to debug issues, adjust configurations, and manage outputs. This structured approach also facilitates collaboration and scalability, as we can easily locate and work with the necessary files and directories.

## 4. The Training Script (`script.sh`)

The `script.sh` file contains the steps required to set up the environment, download necessary data and code, run the training process, and handle periodic checkpointing.

### 4.1 Explanation of `script.sh`

The `script.sh` script is designed to work with the specified directory structure:

- **Variables**: The script defines variables for all the important directories and S3 paths.
- **Sync Operations**: Uses `aws s3 sync` commands to synchronize code and data from S3 to the appropriate local directories.
- **Training Execution**: Points the training script to the correct data and output directories.
- **Checkpoint Handling**: Downloads checkpoints from S3 to resume training if available.
- **Periodic Syncing**: The background process ensures that outputs and checkpoints in the `out` directory are periodically synced to S3.
- **Final Syncing**: After training, the script syncs the `out` directory back to S3, including logs and checkpoints.

Below is the complete `script.sh` file with comments explaining each section.

```bash
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
aws s3 sync "$S3_OUTPUT_DIR/out/" "$LOCAL_OUTPUT_DIR"

# Function to sync outputs and checkpoints periodically to S3
sync_outputs() {
  while true; do
    sleep 600  # Sync every 10 minutes
    echo "Periodic output sync to S3..."
    aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_DIR/out/"
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
aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_OUTPUT_DIR/out/"
echo "Data is saved to S3 at $S3_OUTPUT_DIR/out/"

# Save the log file to S3
aws s3 cp "$LOCAL_CODE_DIR/log.txt" "$S3_OUTPUT_DIR/log.txt"
echo "Log file is saved to S3 at $S3_OUTPUT_DIR/log.txt"
```

#### Key Points:

- **Output Redirection**: All output is logged to `log.txt` for debugging and auditing.

- **Environment Setup**: Uses Conda to create and activate an isolated Python environment.

- **Data Synchronization**: Downloads code and data from Amazon S3 to local directories.

- **Checkpoint Handling**: Syncs checkpoints from S3 to the `out` directory to resume training if available.

- **Periodic Syncing**: Periodically syncs the `out` directory to S3 to prevent data loss.

- **Training Execution**: Runs the training script using `torchrun` for distributed GPU training.

- **Health Signals**: Creates files (`/tmp/ready` and `/tmp/healthy`) to signal readiness and liveness to Kubernetes.

- **Cleanup**: Syncs final outputs and logs back to S3 after training completes.

---

## 5. Kubernetes Job Configuration (`job.yaml`)

The `job.yaml` file defines the Kubernetes Job that orchestrates the training task.

### 5.1 Explanation of `job.yaml`

The `job.yaml` file defines a Kubernetes Job resource, which specifies how to run a batch job on the cluster. Below, we break down the key components of the file and explain their purposes, focusing on `template` and `spec`, as well as including key points for clarity.

Below is the complete `job.yaml` file with comments explaining each section.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: customgpt-job  # Name of the Job
  namespace: dsedge-nogbd  # Namespace where the Job will run
  labels:
    job-name: customgpt-job  # Label for identifying the job
    batch-size: "1200"  # Custom label
    moe-coef: "0.1"     # Custom label
  annotations:
    batch-size: "1200"  # Annotation for metadata
    moe-coef: "0.1"     # Annotation for metadata
spec:
  backoffLimit: 2  # Number of retries before considering the job failed
  template:
    metadata:
      labels:
        job-name: customgpt-job  # Label for the pod template
    spec:
      serviceAccountName: default-editor  # Service account with necessary permissions
      restartPolicy: OnFailure  # Restart policy for the pod
      terminationGracePeriodSeconds: 30  # Time to wait before forceful termination
      volumes:
        - name: user-home  # Volume for user home directory
          emptyDir: {}
        - name: shm  # Shared memory volume
          emptyDir:
            medium: Memory
            sizeLimit: "4Gi"
        - name: script-volume  # Volume for script.sh from ConfigMap
          configMap:
            name: script-configmap
      containers:
        - name: customgpt  # Name of the container
          image: your-docker-image  # Replace with your Docker image
          imagePullPolicy: IfNotPresent  # Pull image only if not present
          resources:
            requests:
              cpu: "4"                # Minimum CPU requested
              memory: "32Gi"          # Minimum memory requested
              nvidia.com/gpu: "4"     # Request 4 GPUs
            limits:
              cpu: "8"                # Maximum CPU allowed
              memory: "64Gi"          # Maximum memory allowed
              nvidia.com/gpu: "4"     # Limit to 4 GPUs
          env:
            # Environment variables for the container
            - name: TARGET
              value: "Python Sample v3"
            - name: S3_BUCKET_SRC
              value: "s3://your-s3-bucket/path/to/code/"
            - name: REPO_NAME
              value: "hello-api"
          volumeMounts:
            # Mount user home directory
            - name: user-home
              mountPath: /home/eai_role
            # Mount shared memory
            - name: shm
              mountPath: /dev/shm
            # Mount script.sh into the container
            - name: script-volume
              mountPath: /home/eai_role/code/script.sh
              subPath: script.sh
              readOnly: true
          command:
            - "/bin/bash"
            - "-c"
            - |
              # Make script.sh executable and run it
              chmod +x /home/eai_role/code/script.sh
              /home/eai_role/code/script.sh
          livenessProbe:
            exec:
              command:
                - cat
                - /tmp/healthy  # Checks for the existence of /tmp/healthy
            initialDelaySeconds: 120   # Wait before starting liveness checks
            periodSeconds: 30          # Time between liveness checks
          readinessProbe:
            exec:
              command:
                - cat
                - /tmp/ready    # Checks for the existence of /tmp/ready
            initialDelaySeconds: 60    # Wait before starting readiness checks
            periodSeconds: 15          # Time between readiness checks
```

#### **Key Components**

##### **1. apiVersion and kind**

Defines the type and version of the Kubernetes resource.

##### **2. metadata**

Provides metadata such as name, namespace, labels, and annotations for organizing and identifying the Job.

##### **3. spec**

Specifies the behavior of the Job, including retry policies and the Pod template.

##### **4. template**

Defines the Pod template used by the Job, specifying the configuration for Pods that the Job creates.

##### **5. template.spec**

Contains the Pod specifications, including containers, volumes, and resource requirements.

##### **6. volumes and volumeMounts**

- **Volumes**: Define storage volumes for the Pod.
- **VolumeMounts**: Mount the volumes into the containers' filesystems.

##### **7. containers**

Defines the containers to run in the Pod, including the image, resources, environment variables, and commands.

##### **8. command**

Specifies the entrypoint command for the container to execute `script.sh`.

##### **9. livenessProbe and readinessProbe**

- **Liveness Probe**: Checks if the container is running properly.
- **Readiness Probe**: Checks if the container is ready to accept traffic.

---

By understanding the purpose and functionality of each component in the `job.yaml` file, your team can confidently modify and deploy Kubernetes Jobs for various applications, ensuring efficient resource utilization and application reliability.

## 6. Deployment Steps

Follow these steps to deploy the training job to your Kubernetes cluster.

### Step 1: Set Up AWS Credentials

Ensure that the Kubernetes cluster or the nodes have access to AWS credentials required for S3 operations.

- **Option 1**: Use IAM roles attached to the nodes.
- **Option 2**: Use Kubernetes Secrets to store AWS credentials and mount them into the Pod.

### Step 2: Update the Docker Image

Replace `your-docker-image` in `job.yaml` with the Docker image that contains the necessary dependencies for your training job.

### Step 3: Create the ConfigMap for `script.sh`

Create a ConfigMap from your `script.sh` file.

```bash
kubectl create configmap script-configmap --from-file=script.sh -n dsedge-nogbd
```

### Step 4: Deploy the Kubernetes Job

Apply the `job.yaml` to deploy the job.

```bash
kubectl apply -f job.yaml
```

### Step 5: Monitor the Job

Check the status of the job and Pods.

```bash
kubectl get jobs -n dsedge-nogbd
kubectl get pods -n dsedge-nogbd
```

View logs to monitor progress.

```bash
kubectl logs <pod-name> -n dsedge-nogbd
```

### Step 6: Verify S3 Synchronization

Ensure that outputs and checkpoints are being synced to your S3 bucket as expected.

### Step 7: Handle Pod Restarts (If Any)

If the Pod restarts due to failure:

- Kubernetes will automatically restart the Pod up to `backoffLimit` times.
- The training script is designed to resume from the last checkpoint.

---

## 7. Conclusion

By following this guide, you can deploy a GPU-accelerated training job on Kubernetes that:

- Utilizes multiple GPUs for intensive computations.

- Ensures data persistence by syncing outputs and checkpoints directly with S3.

- Incorporates health checks for robust monitoring and automatic recovery.

- Leverages Kubernetes features like Jobs, ConfigMaps, and Probes for efficient orchestration.

This setup is scalable and can be adapted to different training tasks or environments. As you become more familiar with Kubernetes, you can further optimize and customize the deployment to suit your specific needs.

---

## Appendix: Additional Resources

- **Kubernetes Documentation**: [https://kubernetes.io/docs/home/](https://kubernetes.io/docs/home/)
- **Kubernetes Jobs**: [https://kubernetes.io/docs/concepts/workloads/controllers/job/](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- **ConfigMaps**: [https://kubernetes.io/docs/concepts/configuration/configmap/](https://kubernetes.io/docs/concepts/configuration/configmap/)
- **GPU Support in Kubernetes**: [https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
