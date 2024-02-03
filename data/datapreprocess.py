from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow as tf
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import math
from skimage.filters import threshold_mean
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import math

def normalise(image):
    # normalise and clip images -1000 to 800
    np_img = image
    np_img = np.clip(np_img, -1000., 800.).astype(np.float32)
    return np_img


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret

def resample(ori_image,target_size=(130,130,80),is_label = False):
    '''
    ori_image:需要被处理的原始图像
    target_size:重采样图像的目标大小
    resample_method:重采样图像时使用的方法（默认为最邻近）

    return:resize后的图像
    '''
    
    resampler = sitk.ResampleImageFilter()
    # 将图像传入重采样器
    resampler.SetReferenceImage(ori_image)
    # 为了防止重采样后图片显示不全，需要更改输出图像的体素大小
    ori_space = ori_image.GetSpacing()
    ori_size = ori_image.GetSize()
    
    #print("original size:",ori_size)
    
    target_space = [ori_size[i] / target_size[i] * ori_space[i] for i in range(len(target_size))]
    target_space = tuple(target_space)
    #print("target space: ",target_space)
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(ori_image.GetOrigin())
    resampler.SetOutputSpacing(target_space)
    resampler.SetOutputDirection(ori_image.GetDirection())

    resampler.SetOutputPixelType(sitk.sitkFloat32)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)


    resampled_image = resampler.Execute(ori_image)
    
    #print("resample size: ",resampled_image.GetSize())
    #print("resample spacing: ",resampled_image.GetSpacing())
    return resampled_image

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def crop_by_boundingbox(image,label):
    
    # Create a binary mask (convert the input grayscale image to binary image)
    binary_mask = sitk.BinaryThreshold(image, lowerThreshold=10, upperThreshold=3000)

    # Calculate the bounding box using LabelStatisticsImageFilter
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(image, binary_mask)

    # Get the bounding box of the binary mask image (the brain region)
    bounding_box = stats_filter.GetBoundingBox(1)
    #print("Bounding Box:", bounding_box) #bounding box: [x_min, x_max, y_min, y_max, z_min, z_max]

    # Adjust the bounding box considering the image spacing
    bb_min_x = int(bounding_box[0])
    # the space between the largest postion of the image and the boundingbox of the brain
    bb_max_x = int((image.GetSize()[0]-bounding_box[1])) 
    bb_min_y = int(bounding_box[2])
    bb_max_y = int((image.GetSize()[1]-bounding_box[3]))
    bb_min_z = int(bounding_box[4])
    bb_max_z = int((image.GetSize()[2]-bounding_box[5]))

    crop_start_index = [bb_min_x, bb_min_y, bb_min_z] # starting point where the cropping start
    crop_size = [bb_max_x, bb_max_y, bb_max_z] # the space to be cropped，上方去除寬度（从右上角图像最大坐标处开始，要裁剪掉的大小）
    # Crop the image, using the starting index and size for cropping
    cropped_image = sitk.Crop(image, crop_start_index, crop_size)
    cropped_label = sitk.Crop(label, crop_start_index, crop_size)
    return cropped_image,cropped_label

def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)

def center_crop(img, size_ratio):
    x, y, z = img.shape
    size_ratio_x, size_ratio_y, size_ratio_z = size_ratio
    size_x = size_ratio_x
    size_y = size_ratio_y
    size_z = size_ratio_z

    if x < size_x or y < size_y and z < size_z:
        raise ValueError

    x1 = 0
    y1 = int((y - size_y) / 2)
    z1 = int((z - size_z) / 2)

    img_crop = img[x1: x1 + size_x, y1: y1 + size_y, z1: z1 + size_z]

    return img_crop

def histogram_matching(mov_scan, ref_scan,
                       histogram_levels=2048,
                       match_points=100,
                       set_th_mean=True):
    """
    Histogram matching following the method developed on
    Nyul et al 2001 (ITK implementation)
    inputs:
    - mov_scan: np.array containing the image to normalize
    - ref_scan np.array containing the reference image
    - histogram levels
    - number of matched points
    - Threshold Mean setting
    outputs:
    - histogram matched image
    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    # perform histogram matching
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref.GetPixelID())

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(histogram_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(set_th_mean)
    matched_vol = matcher.Execute(mov, ref)

    return matched_vol


if __name__ == '__main__':
    
    source_data_root = "D:\\datasets\\20230423_pairs\\MRI"  
    target_data_root = "D:\\datasets\\20230423_pairs\\CT"  
    CT_template_image_root = "D:\\科研\\研零\\多模态配准\\template\\CT.nii.gz"
    CT_template_label_root =  "D:\\科研\\研零\\多模态配准\\template\\CT_label.nii.gz"
    
    # 获取所有子文件夹的名称
    source_subfolders = [subfolder for subfolder in os.listdir(source_data_root)]
    target_subfolders = [subfolder for subfolder in os.listdir(target_data_root)]
    
    for idx in range(118): 
        
        source_subfolder = source_subfolders[idx]
        target_subfolder = target_subfolders[idx]
        #print(source_subfolder)
        # 构建源图像和目标图像的文件路径
        source_path = os.path.join(source_data_root, source_subfolder, "brain.nii.gz")
        source_label_path = os.path.join(source_data_root, source_subfolder, "label.nii.gz")
        #print('source_path: ',source_path)
        target_path = os.path.join(target_data_root, target_subfolder, "brain.nii.gz")
        target_label_path = os.path.join(target_data_root, target_subfolder, "label.nii.gz")
        #print('target_path: ',target_path)
        # 检查文件是否存在
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            raise FileNotFoundError(f"Source or target file not found for subfolder {source_subfolder}")

        source_image_sitk = sitk.ReadImage(source_path)
        target_image_sitk = sitk.ReadImage(target_path)
        source_label_sitk = sitk.ReadImage(source_label_path)
        target_label_sitk = sitk.ReadImage(target_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk) #Converting sitk_metadata to image Array
        target_image_np = sitk.GetArrayFromImage(target_image_sitk)
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)
        target_label_np = sitk.GetArrayFromImage(target_label_sitk)
        
        # Resampling 3D image to spacing=[1, 1, 1]
        source_image_Resampled = resample_img(source_image_sitk, out_spacing=[1, 1, 1], is_label=False)
        target_image_Resampled = resample_img(target_image_sitk, out_spacing=[1, 1, 1], is_label=False)
        source_label_Resampled = resample_img(source_label_sitk, out_spacing=[1, 1, 1], is_label=True)
        target_label_Resampled = resample_img(target_label_sitk, out_spacing=[1, 1, 1], is_label=True)

        
        # Crop background
        source_image_nobackground,source_label_nobackground = crop_by_boundingbox(source_image_Resampled,source_label_Resampled)
        target_image_nobackground,target_label_nobackground = crop_by_boundingbox(target_image_Resampled,target_label_Resampled)
        source_image_nobackground_np = sitk.GetArrayFromImage(source_image_nobackground)
        target_image_nobackground_np = sitk.GetArrayFromImage(target_image_nobackground)
        source_label_nobackground_np = sitk.GetArrayFromImage(source_label_nobackground)
        target_label_nobackground_np = sitk.GetArrayFromImage(target_label_nobackground)
  
        # Pad image to max size
        max_shape = (145,177,169) #这是所有图像三个维度中分别求的最大值
        # 设置填充参数
        kwargs = {'mode': 'constant', 'constant_values': 0}
        source_image_padded = resize_image_with_crop_or_pad(source_image_nobackground_np, max_shape,**kwargs)
        target_image_padded = resize_image_with_crop_or_pad(target_image_nobackground_np, max_shape, **kwargs)
        source_label_padded = resize_image_with_crop_or_pad(source_label_nobackground_np, max_shape,**kwargs)
        target_label_padded = resize_image_with_crop_or_pad(target_label_nobackground_np, max_shape, **kwargs)
        
        # Crop or pad
        source_image_small = resize_image_with_crop_or_pad(source_image_padded, (144, 176, 176),**kwargs)
        target_image_small = resize_image_with_crop_or_pad(target_image_padded,(144, 176, 176), **kwargs)
        source_label_small = resize_image_with_crop_or_pad(source_label_padded, (144, 176, 176),**kwargs)
        target_label_small = resize_image_with_crop_or_pad(target_label_padded, (144, 176, 176), **kwargs)
        
        # Intensity Normalization
        #source_image_IN = normalise_zero_one(source_image_np)
        #target_image_IN = normalise_zero_one(target_image_small)
        
        # Save new image
        #img_list = [source_image_IN,target_image_IN,source_label_small,target_label_small]
        img_list = [source_image_small,source_label_small]
        #resampled_list = [source_image_Resampled,target_image_Resampled,source_label_Resampled,target_label_Resampled]
        new_img_list = []
        #for img_crop in zip(img_list,resampled_list):
        for img_crop in img_list:
            new_image = sitk.GetImageFromArray(img_crop)
            new_image.SetSpacing(source_image_Resampled.GetSpacing())
            new_image.SetDirection(source_image_Resampled.GetDirection())
            new_image.SetOrigin(source_image_Resampled.GetOrigin())
            new_img_list.append(new_image)
            #print("new_img.shape: ",new_image.GetSize())
        

        sitk.WriteImage(new_img_list[0], str(os.path.join(source_data_root, source_subfolder,"brain_small_norigid.nii.gz")))        
        #sitk.WriteImage(new_img_list[1], str(os.path.join(target_data_root, target_subfolder,"brain_small.nii.gz")))
        sitk.WriteImage(new_img_list[1], str(os.path.join(source_data_root, source_subfolder,"label_small_norigid.nii.gz")))
        #sitk.WriteImage(new_img_list[3], str(os.path.join(target_data_root, target_subfolder,"label_small.nii.gz")))
        
        print("Saving image {} succeed".format(source_subfolder[-3:]))