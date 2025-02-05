# brain-tumor-classifier - A Mixture of experts approach

Using pytorch's TIMM package to solve a classification problem

## Details of training
* Training and Testing, separate datasets  found at [kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* Using 2 Experts, with a backbone supported by efficientnet_b2
* Batch of 32 images
* Some image transformations
* Advanced training loops

## MoE Architecture
![MoE Architecture](MoE-Arch.png)


## Results

![Confusion Matrix](confusion_matrix.png)

![Train/Test metrics](train_test_metrics.png)

![Loss](loss.png)
