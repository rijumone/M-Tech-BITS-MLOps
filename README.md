# Assignment-1 [Total Marks - 25]

## M1: MLOps Foundations

**Objective:** Understand the basics of MLOps and implement a simple CI/CD pipeline.

### Tasks:

#### 1. Set Up a CI/CD Pipeline:
- **Tool**: Use a CI/CD tool like GitHub Actions or GitLab CI to set up a pipeline for a sample machine learning project.
- **Stages**:
  - Linting
  - Testing
  - Deploying a simple machine learning model

#### 2. Version Control:
- **Implementation**: Implement version control for your project using Git.
- **Demonstration**: Demonstrate branching, merging, and pull requests.

### Deliverables:
- A report detailing the CI/CD pipeline stages.
- Screenshots or logs showing successful runs of the pipeline.
- A Git repository link with branches and merge history.

## M2: Process and Tooling

**Objective:** Gain hands-on experience with popular MLOps tools and understand the processes they support.

### Tasks:

#### 1. Experiment Tracking:
- **Tool**: Use MLflow to track experiments for a machine learning project.
- **Tracking**: Record metrics, parameters, and results of at least three different model training runs.

#### 2. Data Versioning:
- **Tool**: Use DVC (Data Version Control) to version control a dataset used in your project.
- **Reversion**: Show how to revert to a previous version of the dataset.

### Deliverables:
- MLflow experiment logs with different runs and their results.
- A DVC repository showing different versions of the dataset.

## M3: Model Experimentation and Packaging

**Objective:** Train a machine learning model, perform hyperparameter tuning, and package the model for deployment.

### Tasks:

#### 1. Hyperparameter Tuning:
- **Library**: Use a library like Optuna or Scikit-learn’s GridSearchCV to perform hyperparameter tuning on a chosen model.
- **Documentation**: Document the tuning process and the best parameters found.

#### 2. Model Packaging:
- **Tools**: Package the best-performing model using tools like Docker and Flask.
- **Components**:
  - Create a Dockerfile
  - Develop a simple Flask application to serve the model

### Deliverables:
- A report on hyperparameter tuning results.
- A Dockerfile and Flask application code.
- Screenshots of the model running in a Docker container.

## M4: Model Deployment & Orchestration (Optional)

**Objective:** Deploy a machine learning model and orchestrate its operations using Kubernetes.

### Tasks:

#### 1. Model Deployment:
- **Platform**: Deploy the Dockerized model from M3 to a cloud platform like AWS, Azure, or GCP.
- **Service**: Use a platform service like AWS ECS, Azure AKS, or Google Kubernetes Engine (GKE).

#### 2. Orchestration:
- **Setup**: Set up a Kubernetes cluster.
- **Deployment**: Deploy the model using Kubernetes and create a Helm chart for managing deployments.

### Deliverables:
- A link to the deployed model endpoint.
- Kubernetes configuration files and Helm chart.
- A report detailing the deployment and orchestration process.

## M5: Final Deliverables

- **File**: A zip file containing:
  - Code
  - Data
  - Model

- **Summary**:
  - A one-page summary that includes:
    - Description of the work completed
    - Justification for the choices made

- **Screen Recording** (maximum 5 minutes):
  - Explains the work done.
  - Shows the results.