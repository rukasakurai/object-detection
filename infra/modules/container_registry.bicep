param registryName string
param sku string
param adminUserEnabled bool

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-06-01-preview' = {
  name: registryName
  location: resourceGroup().location
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: adminUserEnabled
  }
}
