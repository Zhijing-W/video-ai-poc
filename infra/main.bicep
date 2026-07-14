// =============================================================
// 视频理解 POC — Azure 基础设施（CPU-first、GPU-ready）
//
// 一次拉起：
//   Resource Group（外部 az group create 建，Bicep 从这里挂进去）
//   ├─ ACR                          镜像仓库
//   ├─ Storage Account (StorageV2)  Blob 容器（models/datasets/results/videos）
//   ├─ Storage Account (Premium)    Azure Files（gallery 共享读写）
//   ├─ Log Analytics + AppInsights  日志/指标
//   ├─ AKS                          CPU node pool（GPU 后加，见 README）
//   └─ 授权：AKS kubelet 可拉 ACR、可读写 Blob、可读写 Files
//
// 设计原则：
//   * managedIdentity 通篇 SystemAssigned；无密钥
//   * standardLB → autoscale 天然支持
//   * cpu pool 允许 autoscale 1–10；GPU pool 后加 `az aks nodepool add`
// =============================================================

@description('部署 region（GPU 配额友好：eastus / eastus2 / westus3 / southcentralus）')
param location string = resourceGroup().location

@description('资源命名前缀，全局唯一（会拼到 ACR/Storage 名字里）')
param namePrefix string = 'videopoc'

@description('CPU 节点 SKU（起步够用；D4s_v5 = 4vCPU/16GB）')
param cpuVmSize string = 'Standard_D4s_v5'

@description('CPU 节点池初始/最小/最大节点数')
param cpuNodeCount int = 2
param cpuMinNodes int = 1
param cpuMaxNodes int = 10

@description('Kubernetes 版本（留空取 AKS 默认稳定版）')
param kubernetesVersion string = ''

@description('给你自己（部署者）加 AKS 管理员角色的 objectId；跑 `az ad signed-in-user show --query id -o tsv` 拿')
param adminObjectId string = ''

// -----------------------------------------------------------------
// 命名（全部小写、去横线，符合 ACR/Storage 命名规范）
// -----------------------------------------------------------------
var suffix = uniqueString(resourceGroup().id)
var acrName        = toLower('${namePrefix}acr${suffix}')
var storageName    = toLower('${namePrefix}st${suffix}')
var filesStoreName = toLower('${namePrefix}fs${suffix}')
var lawName        = '${namePrefix}-law-${suffix}'
var aksName        = '${namePrefix}-aks-${suffix}'
var appInsightsName = '${namePrefix}-ai-${suffix}'

// -----------------------------------------------------------------
// 1) ACR
// -----------------------------------------------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

// -----------------------------------------------------------------
// 2) Blob Storage（models/datasets/results/videos）
// -----------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
    accessTier: 'Hot'
  }
}
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}
resource containerModels 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'models'
  properties: { publicAccess: 'None' }
}
resource containerDatasets 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'datasets'
  properties: { publicAccess: 'None' }
}
resource containerResults 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'results'
  properties: { publicAccess: 'None' }
}
resource containerVideos 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'videos'
  properties: { publicAccess: 'None' }
}

// -----------------------------------------------------------------
// 3) Premium Files（gallery 共享读写；FAISS + K/V 元数据）
// -----------------------------------------------------------------
resource filesStore 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: filesStoreName
  location: location
  sku: { name: 'Premium_LRS' }
  kind: 'FileStorage'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}
resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: filesStore
  name: 'default'
  properties: {
    shareDeleteRetentionPolicy: { enabled: true, days: 7 }
  }
}
resource shareGallery 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: 'gallery'
  properties: {
    shareQuota: 100
    enabledProtocols: 'SMB'
  }
}

// -----------------------------------------------------------------
// 4) Log Analytics + Application Insights
// -----------------------------------------------------------------
resource law 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
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
// 5) AKS — CPU node pool；GPU pool 后加
// -----------------------------------------------------------------
resource aks 'Microsoft.ContainerService/managedClusters@2024-05-01' = {
  name: aksName
  location: location
  identity: { type: 'SystemAssigned' }
  sku: { name: 'Base', tier: 'Free' }
  properties: {
    kubernetesVersion: empty(kubernetesVersion) ? null : kubernetesVersion
    dnsPrefix: aksName
    enableRBAC: true
    aadProfile: {
      managed: true
      enableAzureRBAC: true
    }

    agentPoolProfiles: [
      {
        name: 'systempool'
        mode: 'System'
        vmSize: 'Standard_D2s_v5'
        count: 1
        minCount: 1
        maxCount: 2
        enableAutoScaling: true
        osType: 'Linux'
        type: 'VirtualMachineScaleSets'
      }
      {
        name: 'cpupool'
        mode: 'User'
        vmSize: cpuVmSize
        count: cpuNodeCount
        minCount: cpuMinNodes
        maxCount: cpuMaxNodes
        enableAutoScaling: true
        osType: 'Linux'
        type: 'VirtualMachineScaleSets'
        nodeLabels: { workload: 'app' }
      }
      // GPU pool 现在不建。GPU 配额批下来后运行：
      //   az aks nodepool add -g <RG> --cluster-name <AKS> --name gpupool \
      //     --node-vm-size Standard_NC6ads_A10_v5 --node-count 0 \
      //     --min-count 0 --max-count 3 --enable-cluster-autoscaler \
      //     --priority Spot --spot-max-price -1 \
      //     --node-taints sku=gpu:NoSchedule --labels workload=gpu
    ]

    networkProfile: {
      networkPlugin: 'azure'
      networkPolicy: 'calico'
      loadBalancerSku: 'standard'
      serviceCidr: '10.0.0.0/16'
      dnsServiceIP: '10.0.0.10'
    }

    storageProfile: {
      blobCSIDriver:  { enabled: true }
      diskCSIDriver:  { enabled: true }
      fileCSIDriver:  { enabled: true }
    }
    addonProfiles: {
      omsagent: {
        enabled: true
        config: { logAnalyticsWorkspaceResourceID: law.id }
      }
      azureKeyvaultSecretsProvider: {
        enabled: true
        config: { enableSecretRotation: 'true' }
      }
    }

    autoUpgradeProfile: { upgradeChannel: 'stable' }
  }
}

// -----------------------------------------------------------------
// 6) 授权：AKS kubelet 可拉 ACR、可读写 Blob / Files
// -----------------------------------------------------------------
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'   // AcrPull
var blobOwnerId   = 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'   // Storage Blob Data Owner
var fileContribId = '0c867c2a-1d8c-454a-a3db-ab2ea1bdc8bb'   // Storage File Data SMB Share Contributor

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, aks.id, 'acrPull')
  scope: acr
  properties: {
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalType: 'ServicePrincipal'
  }
}
resource blobOwner 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, aks.id, 'blobOwner')
  scope: storage
  properties: {
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', blobOwnerId)
    principalType: 'ServicePrincipal'
  }
}
resource fileWrite 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(filesStore.id, aks.id, 'fileContrib')
  scope: filesStore
  properties: {
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', fileContribId)
    principalType: 'ServicePrincipal'
  }
}

// 部署者本人 → AKS RBAC Cluster Admin（不指定就用 az aks get-credentials --admin 兜底）
var aksAdminRoleId = 'b1ff04bb-8a4e-4dc4-8eb5-8693973ce19b'
resource userAksAdmin 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(adminObjectId)) {
  name: guid(aks.id, adminObjectId, 'aksAdmin')
  scope: aks
  properties: {
    principalId: adminObjectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', aksAdminRoleId)
    principalType: 'User'
  }
}

// -----------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------
output acrLoginServer string     = acr.properties.loginServer
output acrName string            = acr.name
output aksName string            = aks.name
output storageAccountName string  = storage.name
output filesStoreName string     = filesStore.name
output appInsightsConnStr string = appInsights.properties.ConnectionString
output logAnalyticsId string     = law.id
