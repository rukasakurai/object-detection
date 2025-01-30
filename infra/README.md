# Infrastructure Deployment Guide

This guide provides instructions for deploying Azure resources using Bicep templates.

## Prerequisites

1. **Azure CLI**: Ensure Azure CLI is installed. Follow the [Azure CLI Installation Guide](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
2. **Bicep CLI**: Install the Bicep CLI by running the following command:
   ```bash
   az bicep install
   ```

## Deploying Resources

1. **Navigate to the `infra/` directory**:
   ```bash
   cd infra/
   ```

2. **Deploy resources using the `main.bicep` file**:
   ```bash
   az deployment group create \
     --resource-group <resource-group-name> \
     --template-file main.bicep
   ```

This will provision the following resources:
- An Azure Machine Learning workspace
- A compute cluster for training
- An Azure Container Registry for model storage
- Storage resources
