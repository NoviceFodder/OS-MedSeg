# OS-MedSeg
This is a PyTorch implementation of OS-MedSeg: Distillation Learning Guided by Image Reconstruction for One-Shot Medical Image Segmentation.
## Framework Architecture
![image](https://github.com/NoviceFodder/OS-MedSeg/blob/main/figures/Framework.png)
## Demo
In [test_example.ipynb](https://github.com/NoviceFodder/OS-MedSeg/blob/main/test_example.ipynb), we provide a demo for medecal image segmentation. 

You can easily apply our pre-trained model and the example data we provide to predict the segmentation labels for an image.
## Dataset
We use 2 public MRI datasets: [OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md), [IXI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md) and 1 private CT dataset in our paper. 

We provide some [example](https://github.com/NoviceFodder/OS-MedSeg/tree/main/data) pre-processed images and their corresponding labels in this repository.
## Pre-trained models
We provide our pre-trained models on OASIS, IXI, and CT datasets, which you can use to evaluate medecal image segmentation.

You can download the OASIS pre-trained model here: [Download OS-MedSeg pre-trained model on OASIS](https://drive.google.com/file/d/1zEt8aLy22FMb2lGZnYRT4u2B2cEIeeX4/view?usp=drive_link)

You can download the IXI pre-trained model here: [Download OS-MedSeg pre-trained model on IXI](https://drive.google.com/file/d/1suzlOnUWUMAWyVDIAMNu5I9fC6B1noe2/view?usp=drive_link)

You can download the CT pre-trained model here: [Download OS-MedSeg pre-trained model on CT](https://drive.google.com/file/d/19F-GZ523SAhOq4BjK9--QaC9bFGVgFQ7/view?usp=drive_link)
## Segmentation Results
### Boxplots for IXI 
![image](https://github.com/NoviceFodder/OS-MedSeg/blob/main/figures/IXI-boxplots.png)
### Boxplots for OASIS
![image](https://github.com/NoviceFodder/OS-MedSeg/blob/main/figures/OASIS-boxplots.png)
### Visualization of Segmentation
![image](https://github.com/NoviceFodder/OS-MedSeg/blob/main/figures/intro.png)

You can find more details in our paper.
# Reference
Some codes are referenced from [VoxelMorph](https://github.com/voxelmorph/voxelmorph) and [CLMorph](https://github.com/lihaoliu-cambridge/unsupervised-medical-image-segmentation). Thanks a lot for their great contribution.

