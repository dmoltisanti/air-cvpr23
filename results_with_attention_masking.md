# Results obtained training models with masked attention

Below we report results obtained training all models masking attention for the padding 0 elements
(see [Errata 09/05/2023](README.md#errata-09052023)) - including Action Modifiers, which was also
affected.
For completeness, we include the _Priors_ and _S3D pre-trained_ baselines in the tables below, however these 
results are the same as those reported in the paper as these baselines are not affected by the issue.
Note that results on HowTo100M Adverbs and VATEX Adverbs are not affected as video in these datasets have
fixed duration, thus no padding is used.

## AIR

### Antonyms during training, action labels during testing (Table 2)

|                 |   mAP W   |   mAP M   |   Acc-A   |
|-----------------|:---------:|:---------:|:---------:|
| Priors          |   0.491   |   0.263   |   0.854   |
| S3D pre-trained |   0.389   |   0.173   |   0.735   |
| Act Mod         |   0.509   |   0.251   |   0.857   |
| MLP + Act Mod   |   0.382   |   0.144   | **0.860** |
| CLS             |   0.606   | **0.289** |   0.841   |
| REG – fixed  δ  |   0.554   |   0.193   |   0.837   |
| REG             | **0.613** |   0.244   |   0.847   |


### Antonyms during training, no action labels during testing (Table 3)

|                 |   mAP W   |   mAP M   |   Acc-A   |
|-----------------|:---------:|:---------:|:---------:|
| Priors          |   0.335   |   0.100   |   0.835   |
| S3D pre-trained |   0.425   |   0.177   |   0.702   |
| Act Mod         |   0.356   |   0.115   |   0.835   |
| MLP + Act Mod   |   0.369   |   0.115   |   0.835   |
| CLS             | **0.578** | **0.314** |   0.841   |
| REG – fixed  δ  |   0.554   |   0.193   |   0.837   |
| REG             |   0.520   |   0.213   | **0.844** |


### No antonyms during training, with action labels during testing (Table 4)

|                 |   mAP W   |   mAP M   |
|-----------------|:---------:|:---------:|
| Priors          |   0.491   |   0.263   |
| S3D pre-trained |   0.389   |   0.173   |
| Act Mod         |   0.508   |   0.249   |
| MLP + Act Mod   |   0.383   |   0.140   |
| CLS             |   0.606   |   0.289   |
| REG             | **0.667** | **0.319** |


## ActivityNet Adverbs

### Antonyms during training, action labels during testing (Table 2)

|                 |   mAP W   |   mAP M   |   Acc-A   |
|-----------------|:---------:|:---------:|:---------:|
| Priors          | **0.217** | **0.159** |   0.745   |
| S3D pre-trained |   0.118   |   0.070   |   0.560   |
| Act Mod         |   0.184   |   0.125   | **0.753** |
| MLP + Act Mod   |   0.131   |   0.087   | **0.753** |
| CLS             |   0.130   |   0.096   |   0.741   |
| REG – fixed  δ  |   0.115   |   0.075   |   0.706   |
| REG             |   0.119   |   0.079   |   0.714   |

### Antonyms during training, no action labels during testing (Table 3)

|                 |   mAP W   |   mAP M   |   Acc-A   |
|-----------------|:---------:|:---------:|:---------:|
| Priors          |   0.094   |   0.050   |   0.692   |
| S3D pre-trained |   0.113   |   0.065   |   0.598   |
| Act Mod         |   0.110   |   0.062   |   0.716   |
| MLP + Act Mod   |   0.110   |   0.062   |   0.714   |
| CLS             | **0.129** | **0.096** | **0.741** |
| REG – fixed  δ  |   0.114   |   0.075   |   0.706   |
| REG             |   0.120   |   0.079   |   0.716   |

### No antonyms during training, with action labels during testing (Table 4)

|                 |   mAP W   |   mAP M   |
|-----------------|:---------:|:---------:|
| Priors          | **0.217** | **0.159** |
| S3D pre-trained |   0.118   |   0.071   |
| Act Mod         |   0.187   |   0.127   |
| MLP + Act Mod   |   0.136   |   0.090   |
| CLS             |   0.130   |   0.096   |
| REG             |   0.143   |   0.093   |

## MSR-VTT Adverbs

### Antonyms during training, action labels during testing (Table 2)

|                 |   mAP W   |   mAP M   |   Acc-A   |
|-----------------|:---------:|:---------:|:---------:|
| Priors          | **0.308** | **0.152** |   0.723   |
| S3D pre-trained |   0.194   |   0.075   |   0.603   |
| Act Mod         |   0.233   |   0.127   |   0.731   |
| MLP + Act Mod   |   0.184   |   0.123   |   0.731   |
| CLS             |   0.305   |   0.131   |   0.751   |
| REG – fixed  δ  |   0.203   |   0.100   |   0.706   |
| REG             |   0.282   |   0.114   | **0.774** |

### Antonyms during training, no action labels during testing (Table 3)

|                 |   mAP W   |   mAP M   |   Acc-A   |
|-----------------|:---------:|:---------:|:---------:|
| Priors          |   0.137   |   0.056   |   0.723   |
| S3D pre-trained |   0.199   |   0.088   |   0.603   |
| Act Mod         |   0.164   |   0.071   |   0.723   |
| MLP + Act Mod   |   0.163   |   0.080   |   0.723   |
| CLS             | **0.304** |   0.131   |   0.754   |
| REG – fixed  δ  |   0.204   |   0.096   |   0.709   |
| REG             |   0.276   | **0.133** | **0.774** |

### No antonyms during training, with action labels during testing (Table 4)

|                 |   mAP W   |   mAP M   |
|-----------------|:---------:|:---------:|
| Priors          | **0.308** | **0.152** |
| S3D pre-trained |   0.194   |   0.075   |
| Act Mod         |   0.233   |   0.134   |
| MLP + Act Mod   |   0.193   |   0.122   |
| CLS             |   0.305   |   0.131   |
| REG             |   0.287   |   0.121   |