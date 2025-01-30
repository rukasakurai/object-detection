param workspaceName string
param location string

resource workspace 'Microsoft.MachineLearningServices/workspaces@2021-04-01' = {
  name: workspaceName
  location: location
  properties: {}
}
