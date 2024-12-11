# Object Detection Tutorial

This repository demonstrates how to train and deploy an object detection model using PyTorch and Azure. Below are the steps to leverage the code for training and inference.

---

## **Steps to Get Started**

### **1. Prepare Training, Validation, and Test Data**

1. **Purpose of Each Dataset**:
   - **Training Data:** Used to train the model. Contains labeled examples to teach the model how to detect objects.
   - **Validation Data:** Used to tune hyperparameters and evaluate the model's performance during training. This ensures the model generalizes well to unseen data.
   - **Test Data:** Held back until final evaluation. Used to measure the model's accuracy on completely unseen data.

2. **Organize Data**:
   - Place raw images in the `data/raw/images/` directory.
   - Annotate the data using a labeling tool like LabelImg and save the annotations in the `data/labeled/annotations/` directory.
   - Split the data into training, validation, and test sets:
     - Save processed training data in `data/processed/train/`.
     - Save validation data in `data/processed/val/`.
     - Save test data in `data/processed/test/`.

3. **Annotation Format**:
   - Use a standard format like COCO or YOLO for consistency with the model.

---

### **2. Provision Necessary Cloud Resources**

Use the provided Bicep templates to provision Azure resources for training and deployment:

1. **Install Azure CLI and Bicep**:
   - Ensure Azure CLI is installed: [Azure CLI Installation Guide](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
   - Install the Bicep CLI:
     ```bash
     az bicep install
     ```

2. **Deploy Resources**:
   - Navigate to the `infra/` directory.
   - Use the `main.bicep` file to deploy all necessary resources:
     ```bash
     az deployment group create \
       --resource-group <resource-group-name> \
       --template-file main.bicep
     ```
   - This will provision:
     - An Azure Machine Learning workspace
     - A compute cluster for training
     - An Azure Container Registry for model storage

---

### **3. Train the Model**

1. **Setup Environment**:
   - Create a virtual environment and install dependencies:
     ```bash
     python -m venv .venv
     source .venv/bin/activate   # On Windows: .venv\Scripts\activate
     pip install -r requirements.txt
     ```

2. **Train the Model**:
   - Open the `notebooks/training.ipynb` notebook.
   - Update the dataset paths to point to your prepared training and validation data.
   - Train the model on the Azure compute cluster by executing the notebook.

3. **Save the Trained Model to Azure Container Registry**:
   - **Steps to Save**:
     1. Create a Docker image containing the trained model:
        ```bash
        docker build -t <acr_name>.azurecr.io/trained_model:latest .
        ```
     2. Push the Docker image to the Azure Container Registry:
        ```bash
        az acr login --name <acr_name>
        docker push <acr_name>.azurecr.io/trained_model:latest
        ```
     3. The model is now stored in ACR and ready for deployment.

---

### **4. Perform Inference**

1. **Run Inference Locally**:
   - Open the `notebooks/inference.ipynb` notebook.
   - Load the trained model from Azure Container Registry.
   - Perform inference on test images and visualize the results.

2. **Run Inference in the Cloud**:
   - Deploy the model as a web service to Azure Container Apps.
   - Update the notebook or scripts to send test images to the deployed endpoint and retrieve predictions.

---

### **Dependencies**
Ensure you have the required Python libraries installed as specified in `requirements.txt`.
