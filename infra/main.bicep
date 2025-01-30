targetScope = 'resourceGroup'

module workspace 'modules/workspace.bicep' = {
  name: 'workspaceDeployment'
  params: {
    workspaceName: 'myWorkspace'
    location: resourceGroup().location
  }
}

module computeCluster 'modules/compute_cluster.bicep' = {
  name: 'computeClusterDeployment'
  params: {
    clusterName: 'myComputeCluster'
    vmSize: 'Standard_D2_v2'
    minNodes: 0
    maxNodes: 4
  }
}

module containerRegistry 'modules/container_registry.bicep' = {
  name: 'containerRegistryDeployment'
  params: {
    registryName: 'myContainerRegistry'
    sku: 'Standard'
    adminUserEnabled: true
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storageDeployment'
  params: {
    storageAccountName: 'mystorageaccount'
    sku: 'Standard_LRS'
  }
}
