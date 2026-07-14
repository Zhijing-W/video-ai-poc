// =============================================================
// 视频理解 POC — Azure 基础设施（无 AKS 版）
//
// AKS 在 Portal 上手动创建，Bicep 只建基础设施：
//   ├─ ACR                          镜像仓库
//   ├─ Storage Account (StorageV2)  Blob 容器（models/datasets/results/videos）
//   ├─ Storage Account (Premium)    Azure Files（gallery 共享读写）
//   └─ Log Analytics + AppInsights  日志/指标
//
// AKS 建好后手动加授权（见 infra/grant-aks-access.ps1）
// =============================================================

@description('部署 region')
param location string = resourceGroup().location

@description('资源命名前缀')
param namePrefix string = 'videopoc'

// -----------------------------------------------------------------
// 命名
// -----------------------------------------------------------------
var suffix = uniqueString(resourceGroup().id)
var acrName        = toLower('${namePrefix}acr${suffix}')
var storageName    = toLower('${namePrefix}st${suffix}')
var filesStoreName = toLower('${namePrefix}fs${suffix}')
var lawName        = '${namePrefix}-law-${suffix}'
var appInsightsName = '${namePrefix}-ai-${suffix}'

// -----------------------------------------------------------------
// 1) ACR
// -----------------------------------------------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
    anonymousPullEnabled: false
  }
}

// -----------------------------------------------------------------
// 2) Blob Storage (标准 StorageV2) + 4 个 container
// -----------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}
resource blobSvc 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}
resource cModels    'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobSvc
  name: 'models'
  properties: { publicAccess: 'None' }
}
resource cDatasets  'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobSvc
  name: 'datasets'
  properties: { publicAccess: 'None' }
}
resource cResults   'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobSvc
  name: 'results'
  properties: { publicAccess: 'None' }
}
resource cVideos    'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobSvc
  name: 'videos'
  properties: { publicAccess: 'None' }
}

// -----------------------------------------------------------------
// 3) Azure Files Premium（gallery 用；POSIX/多 pod RWX）
// -----------------------------------------------------------------
resource filesStore 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: filesStoreName
  location: location
  kind: 'FileStorage'
  sku: { name: 'Premium_LRS' }
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    azureFilesIdentityBasedAuthentication: { directoryServiceOptions: 'None' }
  }
}
resource filesSvc 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: filesStore
  name: 'default'
}
resource galleryShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: filesSvc
  name: 'gallery'
  properties: {
    shareQuota: 100    // GiB
    enabledProtocols: 'SMB'
  }
}

// -----------------------------------------------------------------
// 4) Log Analytics + App Insights
// -----------------------------------------------------------------
resource law 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: lawName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: law.id
  }
}

// -----------------------------------------------------------------
// Outputs (AKS 建好后手动加 role assignments 用)
// -----------------------------------------------------------------
output acrLoginServer string     = acr.properties.loginServer
output acrName string            = acr.name
output acrResourceId string      = acr.id
output storageAccountName string  = storage.name
output storageResourceId string  = storage.id
output filesStoreName string     = filesStore.name
output filesResourceId string    = filesStore.id
output appInsightsConnStr string = appInsights.properties.ConnectionString
output logAnalyticsId string     = law.id
