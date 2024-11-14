# Report: Deploying a GPU-Accelerated Training Job on Kubernetes

## Introduction

This report provides a step-by-step guide to deploying a GPU-accelerated machine learning training job on Kubernetes. It covers the following:

- **Understanding the Kubernetes Job and its components**
- **Explaining the `script.sh` file used for training**
- **Detailed walkthrough of the Kubernetes YAML configuration**
- **Instructions on how to deploy the job to a Kubernetes cluster**

By the end of this report, we should have a clear understanding of how to set up and run a GPU-intensive training task on Kubernetes using the provided configurations.

---

## Table of Contents

1. [Overview of Kubernetes Concepts](#1-overview-of-kubernetes-concepts)
2. [File Layout and Directory Structure](#2-file-layout-and-directory-structure)
   - 2.1 [Files in Amazon S3](#21-files-in-amazon-s3)
   - 2.2 [Directory Structure in the Kubernetes Pod](#22-directory-structure-in-the-kubernetes-pod)
3. [The Training Script (`script.sh`)](#3-the-training-script-scriptsh)
   - 3.1 [Explanation of `script.sh`](#31-explanation-of-scriptsh)
4. [Kubernetes Job Configuration (`job.yaml`)](#4-kubernetes-job-configuration-jobyaml)
   - 4.1 [Explanation of `job.yaml`](#41-explanation-of-jobyaml)
5. [Deployment Steps](#5-deployment-steps)
6. [Conclusion](#6-conclusion)

---

## 1. Overview of Kubernetes Concepts

Before diving into the configurations, let's briefly cover some Kubernetes concepts relevant to this deployment.

### Kubernetes Components Used:

- **Pod**: The smallest deployable unit in Kubernetes, which can contain one or more containers.
- **Job**: A controller that creates one or more pods to run a finite task to completion.
- **Container**: An instance of a Docker image running within a pod.
- **ConfigMap**: A Kubernetes object to store non-confidential configuration data in key-value pairs.
- **Volume**: A directory, possibly with data in it, accessible to the containers in a pod.
- **Resource Requests and Limits**: Specifications to inform Kubernetes about the minimum and maximum resources (CPU, memory, GPU) a container needs.
- **Probes (Liveness and Readiness)**: Mechanisms to check the health and readiness of a container.

---

## 2. File Layout and Directory Structure

Understanding the organization of files and directories is crucial for managing the code, data, and outputs of your training job. This section details the file layout both in Amazon S3 and within the Kubernetes pod after copying the files. It also explains how outputs such as logs and checkpoints are stored.

### 2.1 Files in Amazon S3

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
- **job.yaml**: Kubernetes Job configuration file (used locally, not in the pod).

#### **Data Directory Structure**

Your data folder in Amazon S3 contains the training and validation datasets:

```
s3://your-s3-bucket/path/to/data/
├── train.bin
└── val.bin
```

- **train.bin**: Serialized training data.
- **val.bin**: Serialized validation data.

### 2.2 Directory Structure in the Kubernetes Pod

After copying the files from Amazon S3 to the pod, the directory structure within the container is organized as follows:

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

By understanding the file layout and directory structure, we can better navigate the environment within the Kubernetes pod, making it easier to debug issues, adjust configurations, and manage outputs. This structured approach also facilitates collaboration and scalability, as we can easily locate and work with the necessary files and directories.

## 3. The Training Script (`script.sh`)

The `script.sh` file contains the steps required to set up the environment, download necessary data and code, run the training process, and handle periodic checkpointing.

### 3.1 Explanation of `script.sh`

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
- **Environment Setup**: Uses conda to create and activate an isolated Python environment.
- **Data Synchronization**: Downloads code and data from Amazon S3 to local directories.
- **Checkpoint Handling**: Syncs checkpoints from S3 to the `out` directory to resume training if available.
- **Periodic Syncing**: Periodically syncs the `out` directory to S3 to prevent data loss.
- **Training Execution**: Runs the training script using `torchrun` for distributed GPU training.
- **Health Signals**: Creates files (`/tmp/ready` and `/tmp/healthy`) to signal readiness and liveness to Kubernetes.
- **Cleanup**: Syncs final outputs and logs back to S3 after training completes.

---

## 4. Kubernetes Job Configuration (`job.yaml`)

The `job.yaml` file defines the Kubernetes Job that orchestrates the training task.

### 4.1 Explanation of `job.yaml`

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

```yaml
apiVersion: batch/v1
kind: Job
```

- **apiVersion**: Specifies the version of the Kubernetes API to use. `batch/v1` indicates that we're using version 1 of the batch API, which includes Job resources.
- **kind**: Defines the type of Kubernetes resource. In this case, it's a `Job`, which is used for running finite tasks to completion.

##### **2. metadata**

```yaml
metadata:
  name: customgpt-job
  namespace: dsedge-nogbd
  labels:
    job-name: customgpt-job
    batch-size: "1200"
    moe-coef: "0.1"
  annotations:
    batch-size: "1200"
    moe-coef: "0.1"
```

- **name**: A unique identifier for the Job within the namespace.
- **namespace**: The namespace in which the Job is deployed. Namespaces provide a way to divide cluster resources between multiple users or teams.
- **labels**: Key-value pairs used for organizing, categorizing, and selecting resources. They are useful for filtering and querying Kubernetes objects.
- **annotations**: Key-value pairs used to attach arbitrary non-identifying metadata to objects. They can be used by tools and libraries to store additional information.

**Key Point**: Metadata helps in organizing and managing resources within the Kubernetes cluster, enabling easy identification and selection.

##### **3. spec**

```yaml
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        job-name: customgpt-job
    spec:
      serviceAccountName: default-editor
      restartPolicy: OnFailure
      terminationGracePeriodSeconds: 30
      volumes:
        # Volume definitions
      containers:
        # Container definitions
```

- **spec**: The specification of the Job, detailing how the Job should be executed. It includes configurations such as retries, pod templates, and scheduling details.
- **backoffLimit**: The number of retries before the Job is considered failed. In this example, Kubernetes will attempt to run the Job 3 times (initial try + 2 retries) before marking it as failed.

**Key Point**: The `spec` defines the behavior and execution details of the Job, controlling aspects like retries and the pod template.

##### **4. template**

Within the Job's `spec`, the `template` field is a **Pod template** that describes the pods that will be created by the Job.

```yaml
template:
  metadata:
    labels:
      job-name: customgpt-job
  spec:
    # Pod specifications
```

- **template**: This field contains the pod template, which is essentially a blueprint for the pods that the Job controller will create to execute the task.
  - **metadata**: Contains labels and annotations for the pods created from this template.
  - **spec**: The pod specification, defining containers, volumes, and other settings.

**Purpose of `template`**:

- **Defines Pod Configuration**: The `template` specifies how the pods should be configured, including containers, volumes, and environment variables.
- **Ensures Consistency**: All pods created by the Job will be identical and conform to the template specifications.

**Key Point**: The `template` field within the Job's `spec` defines the pod configuration that will be used to create the pods running the Job.

##### **5. template.spec**

The `spec` within the `template` (referred to as `template.spec`) contains the actual pod specifications.

```yaml
spec:
  serviceAccountName: default-editor
  restartPolicy: OnFailure
  terminationGracePeriodSeconds: 30
  volumes:
    # Volume definitions
  containers:
    # Container definitions
```

- **serviceAccountName**: Specifies the service account under which the pod runs, providing it with necessary permissions.
- **restartPolicy**: Determines when the containers within the pod should be restarted. `OnFailure` means containers will restart on failure.
- **terminationGracePeriodSeconds**: Time given to the pod to terminate gracefully before being forcefully killed.
- **volumes**: Defines storage volumes to be used by the containers.
- **containers**: Specifies the containers that will run in the pod, including images, resources, and commands.

**Purpose of `template.spec`**:

- **Pod Behavior and Configuration**: Defines how the pod operates, including resource allocation, container images, commands, and volume mounts.
- **Container Definitions**: Details the containers to run within the pod, including their specific configurations.

**Key Point**: `template.spec` is critical as it outlines the operational parameters and configuration of the pods created by the Job.

##### **6. volumes and volumeMounts**

Volumes are defined at the pod level and are used to share data between containers or provide necessary filesystem resources.

```yaml
volumes:
  - name: user-home
    emptyDir: {}
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: "4Gi"
  - name: script-volume
    configMap:
      name: script-configmap
```

- **user-home**: An empty directory that is created for the pod; data is lost when the pod is deleted.
- **shm**: An in-memory volume used for shared memory; beneficial for certain applications like PyTorch.
- **script-volume**: A volume that injects the `script.sh` file from a ConfigMap into the container.

**VolumeMounts in Containers**:

```yaml
volumeMounts:
  - name: user-home
    mountPath: /home/eai_role
  - name: shm
    mountPath: /dev/shm
  - name: script-volume
    mountPath: /home/eai_role/code/script.sh
    subPath: script.sh
    readOnly: true
```

- **mountPath**: Specifies where the volume is mounted within the container's filesystem.
- **subPath**: Allows mounting a single file from the volume (useful for ConfigMaps).
- **readOnly**: Ensures the volume is mounted as read-only.

**Key Point**: Volumes and volume mounts are essential for data sharing between containers and injecting configuration files into containers.

##### **7. containers**

Defines the containers that run within the pod.

```yaml
containers:
  - name: customgpt
    image: your-docker-image
    imagePullPolicy: IfNotPresent
    resources:
      requests:
        cpu: "4"
        memory: "32Gi"
        nvidia.com/gpu: "4"
      limits:
        cpu: "8"
        memory: "64Gi"
        nvidia.com/gpu: "4"
    env:
      # Environment variables
    volumeMounts:
      # Volume mounts
    command:
      # Commands to execute
    livenessProbe:
      # Liveness probe settings
    readinessProbe:
      # Readiness probe settings
```

- **name**: Name of the container.
- **image**: Docker image to use.
- **imagePullPolicy**: Determines when Kubernetes should pull the image.
- **resources**:
  - **requests**: Minimum resources needed for the container (used for scheduling).
  - **limits**: Maximum resources the container can use.
- **env**: Environment variables passed to the container.
- **volumeMounts**: Specifies volumes to mount into the container's filesystem.
- **command**: The command to execute when the container starts.
- **livenessProbe**: Checks if the container is running properly.
- **readinessProbe**: Checks if the container is ready to accept traffic.

**Key Points**:

- **Resource Management**: Setting resource requests and limits helps Kubernetes schedule the pod efficiently and prevents resource contention.
- **GPU Allocation**: By specifying `nvidia.com/gpu`, you ensure the pod is scheduled on a node with the required GPUs.
- **Environment Variables**: Useful for passing configuration data to the application running inside the container.
- **Health Probes**: Liveness and readiness probes improve the reliability of applications by enabling Kubernetes to detect and handle unhealthy containers.

##### **8. command**

Specifies the entrypoint command for the container.

```yaml
command:
  - "/bin/bash"
  - "-c"
  - |
    # Make script.sh executable and run it
    chmod +x /home/eai_role/code/script.sh
    /home/eai_role/code/script.sh
```

- **Purpose**: Executes the `script.sh` file, which contains the logic for setting up the environment, downloading data, and running the training job.

**Key Point**: Defining the command at the container level allows for dynamic execution of scripts and commands without baking them into the Docker image.

##### **9. livenessProbe and readinessProbe**

Used by Kubernetes to monitor the health and readiness of the container.

```yaml
livenessProbe:
  exec:
    command:
      - cat
      - /tmp/healthy
  initialDelaySeconds: 120
  periodSeconds: 30

readinessProbe:
  exec:
    command:
      - cat
      - /tmp/ready
  initialDelaySeconds: 60
  periodSeconds: 15
```

- **livenessProbe**:
  - Checks for the existence of `/tmp/healthy` to determine if the container is healthy.
  - If the probe fails, Kubernetes restarts the container.
- **readinessProbe**:
  - Checks for the existence of `/tmp/ready` to determine if the container is ready to accept requests.
  - Helps ensure that traffic is not sent to a container before it is fully ready.

**Key Point**: Health probes are essential for building resilient applications that can recover from failures and ensure consistent service availability.

---

#### **Summary of Key Points**

- **Job Resource**: Orchestrates the execution of pods for batch processing tasks.
- **Template and Spec**:
  - **template**: Defines the pod template used by the Job to create pods.
  - **spec**: Within the template, specifies the configuration and behavior of the pods.
- **Resource Allocation**: Properly specifying resource requests and limits ensures efficient scheduling and resource utilization.
- **Configuration Management**: ConfigMaps and environment variables provide flexible configuration without modifying container images.
- **Command Execution**: Custom commands allow for dynamic execution and flexibility in running scripts.
- **Health Monitoring**: Liveness and readiness probes enable Kubernetes to manage container health and readiness effectively.

---

By understanding the purpose and functionality of each component in the `job.yaml` file, your team can confidently modify and deploy Kubernetes Jobs for various applications, ensuring efficient resource utilization and application reliability.

## 5. Deployment Steps

Follow these steps to deploy the training job to your Kubernetes cluster.

### Step 1: Set Up AWS Credentials

Ensure that the Kubernetes cluster or the nodes have access to AWS credentials required for S3 operations.

- **Option 1**: Use IAM roles attached to the nodes.
- **Option 2**: Use Kubernetes Secrets to store AWS credentials and mount them into the pod.

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

Check the status of the job and pods.

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

If the pod restarts due to failure:

- Kubernetes will automatically restart the pod up to `backoffLimit` times.
- The training script is designed to resume from the last checkpoint.

---

## 6. Conclusion

By following this guide, you can deploy a GPU-accelerated training job on Kubernetes that:

- Utilizes multiple GPUs for intensive computations.
- Ensures data persistence by syncing outputs and checkpoints directly with S3.
- Incorporates health checks for robust monitoring and automatic recovery.
- Leverages Kubernetes features like Jobs, ConfigMaps, and Probes for efficient orchestration.

This setup is scalable and can be adapted to different training tasks or environments. As we becomes more familiar with Kubernetes, you can further optimize and customize the deployment to suit your specific needs.

---

## Appendix: Additional Resources

- **Kubernetes Documentation**: [https://kubernetes.io/docs/home/](https://kubernetes.io/docs/home/)
- **Kubernetes Jobs**: [https://kubernetes.io/docs/concepts/workloads/controllers/job/](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- **ConfigMaps**: [https://kubernetes.io/docs/concepts/configuration/configmap/](https://kubernetes.io/docs/concepts/configuration/configmap/)
- **GPU Support in Kubernetes**: [https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)