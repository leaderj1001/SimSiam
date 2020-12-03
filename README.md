# Exploring Simple Siamese Representation Learning (SimSiam)

## Network

<img width="468" alt="simsiam" src="https://user-images.githubusercontent.com/22078438/100966878-4a5b3500-3571-11eb-967e-e9171448f0e6.png">

  
## Experiments
  | Model | Pre-training Epochs | Batch size | Dim | Linear Evaluation | Acc (%) |
  |:-:|:-:|:-:|:-:|:-:|:-:|
  | ResNet-18 (Paper) | 800 | 512 | 2048 | O | 91.8 |
  | ResNet-18 (Our) | 300 | 512 | 1024 | O | 72.49 |

## Usage
  - Dataset (CIFAR-10)
    - [Data Link](https://www.cs.toronto.edu/~kriz/cifar.html)
    ```
    data
      └── cifar-10-batches-py
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        ├── readme.html
        └── test_batch
    ```
  1. Pre-training
  ```
  python main.py --pretrain True
  ```
  
  2. DownStream Task (Linear)
  ```
  python main.py --checkpoints checkpoints/checkpoint_pretrain_model.pth --pretrain False
  ```

## Reference
  - [Paper Link](https://arxiv.org/abs/2011.10566)
  - Author: Xinlei Chen, Kaiming He
  - Organization: Facebook AI Research (FAIR)
