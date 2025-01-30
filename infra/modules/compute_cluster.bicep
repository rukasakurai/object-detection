param clusterName string
param vmSize string
param minNodes int
param maxNodes int

resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes' = {
  name: clusterName
  location: resourceGroup().location
  properties: {
    vmSize: vmSize
    scaleSettings: {
      minNodeCount: minNodes
      maxNodeCount: maxNodes
    }
  }
}
