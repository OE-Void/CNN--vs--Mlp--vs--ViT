## Comparing the most promising architecture for AI image processing of the decade

This repo have a the code and assets related to comaprison of the top 3 best archiecture used widely for image related tasks (eg. image-detecttion, image -classifiaticon etc) 

we will be rabnking all 3 arctitectures on the basis of speed for training and accuracy on a given tasks with a ceritain size dataset 

**info**

- dataset used :  Mnist

- Train-Split size:  60k samples

- Test-Split size:  10k samples 
## results
```Bash
#Terminal
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

========== Training MLP ==========
Epoch 1/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 9s 10ms/step - accuracy: 0.8590 - loss: 0.4684 - val_accuracy: 0.9603 - val_loss: 0.1329
Epoch 2/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9636 - loss: 0.1152 - val_accuracy: 0.9707 - val_loss: 0.0954
Epoch 3/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9754 - loss: 0.0809 - val_accuracy: 0.9749 - val_loss: 0.0826
Epoch 4/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9808 - loss: 0.0598 - val_accuracy: 0.9762 - val_loss: 0.0819
Epoch 5/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9817 - loss: 0.0551 - val_accuracy: 0.9806 - val_loss: 0.0674
Epoch 6/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - accuracy: 0.9861 - loss: 0.0438 - val_accuracy: 0.9822 - val_loss: 0.0656
Epoch 7/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9890 - loss: 0.0369 - val_accuracy: 0.9805 - val_loss: 0.0723
Epoch 8/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9884 - loss: 0.0359 - val_accuracy: 0.9810 - val_loss: 0.0675
Epoch 9/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9893 - loss: 0.0331 - val_accuracy: 0.9795 - val_loss: 0.0802
Epoch 10/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9896 - loss: 0.0319 - val_accuracy: 0.9801 - val_loss: 0.0687

========== Training CNN ==========
Epoch 1/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 19ms/step - accuracy: 0.8503 - loss: 0.4694 - val_accuracy: 0.9846 - val_loss: 0.0472
Epoch 2/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 12s 9ms/step - accuracy: 0.9807 - loss: 0.0637 - val_accuracy: 0.9896 - val_loss: 0.0286
Epoch 3/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9870 - loss: 0.0426 - val_accuracy: 0.9910 - val_loss: 0.0255
Epoch 4/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9909 - loss: 0.0293 - val_accuracy: 0.9912 - val_loss: 0.0254
Epoch 5/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9933 - loss: 0.0220 - val_accuracy: 0.9936 - val_loss: 0.0195
Epoch 6/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.9946 - loss: 0.0173 - val_accuracy: 0.9920 - val_loss: 0.0212
Epoch 7/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9946 - loss: 0.0170 - val_accuracy: 0.9934 - val_loss: 0.0198
Epoch 8/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9953 - loss: 0.0147 - val_accuracy: 0.9928 - val_loss: 0.0201
Epoch 9/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.9962 - loss: 0.0116 - val_accuracy: 0.9932 - val_loss: 0.0243
Epoch 10/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9965 - loss: 0.0124 - val_accuracy: 0.9935 - val_loss: 0.0197

========== Training ViT ==========
Epoch 1/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 64s 63ms/step - accuracy: 0.6262 - loss: 1.1094 - val_accuracy: 0.9366 - val_loss: 0.2167
Epoch 2/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9323 - loss: 0.2190 - val_accuracy: 0.9619 - val_loss: 0.1264
Epoch 3/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.9576 - loss: 0.1331 - val_accuracy: 0.9687 - val_loss: 0.1016
Epoch 4/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9660 - loss: 0.1087 - val_accuracy: 0.9759 - val_loss: 0.0820
Epoch 5/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9709 - loss: 0.0942 - val_accuracy: 0.9788 - val_loss: 0.0729
Epoch 6/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9722 - loss: 0.0879 - val_accuracy: 0.9773 - val_loss: 0.0721
Epoch 7/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9758 - loss: 0.0749 - val_accuracy: 0.9775 - val_loss: 0.0731
Epoch 8/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.9778 - loss: 0.0695 - val_accuracy: 0.9792 - val_loss: 0.0717
Epoch 9/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9788 - loss: 0.0670 - val_accuracy: 0.9797 - val_loss: 0.0694
Epoch 10/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - accuracy: 0.9809 - loss: 0.0584 - val_accuracy: 0.9805 - val_loss: 0.0587

Benchmark Summary:
--------------------------------------------------
MLP: Params=932,362 | Time=29.8s | Final Val Acc=0.9801
CNN: Params=467,818 | Time=59.9s | Final Val Acc=0.9935
ViT: Params=504,330 | Time=136.1s | Final Val Acc=0.9805
```
---

## Benchmark Analysis

After training all three architectures (MLP, CNN, ViT) on the **MNIST dataset** (60k training samples, 10k test samples), the following observations were made:


- **CNN (Convolutional Neural Network)**  
  - Moderate training time (~60s).  
  - Achieved the **highest accuracy** (~99.35%).  
  - CNNs excel at image tasks because they exploit local spatial patterns (edges, textures, shapes).  
  - Best suited for small to medium‑scale image datasets like MNIST.

- **MLP (Multi‑Layer Perceptron)**  
  - Fastest training time (~30s).  
  - Achieved ~98% validation accuracy.  
  - Good baseline, but limited in capturing spatial features of images.

- **ViT (Vision Transformer)**  
  - Longest training time (~136s).  
  - Achieved ~98.05% validation accuracy, slightly lower than CNN.  
  - **Reason**: Vision Transformers are designed for **large‑scale datasets** (e.g., ImageNet with millions of samples).  
    - They rely heavily on massive amounts of data to learn effective attention patterns.  
    - On smaller datasets like MNIST (only 60k samples), ViTs tend to underperform compared to CNNs.  
  - Despite lower accuracy here, ViTs shine when trained on huge datasets with diverse image structures.

---

## Conclusion

- **Winner on MNIST**: **CNN** — best balance of speed and accuracy.  
- **MLP**: Lightweight and fast, but less accurate.  
- **ViT**: Powerful architecture for large‑scale image datasets, but not optimal for MNIST due to limited sample size.  

---

## Summary Table

| Architecture | Params   | Training Time | Final Val Accuracy | Notes |
|--------------|----------|---------------|--------------------|-------|
| **MLP**      | 932,362  | 29.8s         | 98.01%             | Fast baseline, weaker spatial learning |
| **CNN**      | 467,818  | 59.9s         | **99.35%**         | Best performer on MNIST |
| **ViT**      | 504,330  | 136.1s        | 98.05%             | Needs larger datasets to shine |

