from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
import os
import sys
import glob
from utils import pkload
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.transforms import apply_transform


class CT_MRI_Dataset(Dataset):
    def __init__(self, source_root_dir,target_root_dir,transform2img=None,transform2both=None,is_Training=True,is_val=False):
        
        self.source_root_dir = source_root_dir
        self.target_root_dir = target_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        if is_Training:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][:70]
            self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)][:70]
        elif is_val:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][70:88]
            self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)][70:88]
        else:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][88:]
            self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)][88:]
    
    def __len__(self):
        return len(self.source_subfolders) 
        
    def __getitem__(self, idx):
        
        source_subfolder = self.source_subfolders[idx]
        #target_subfolder = self.target_subfolders[idx]
        #print(source_subfolder)
        # 构建源图像和目标图像的文件路径
        '''
        source_path = os.path.join(self.source_root_dir, source_subfolder, "brain_small_norigid_norm.nii.gz")
        syn_ct_label_path = os.path.join(self.source_root_dir, source_subfolder, "label_small_norigid.nii.gz")
        '''
        #先跑一次不执行rigid预处理的vxm
        source_path = "./dataset/template/CT_small_norm.nii.gz"
        syn_ct_label_path = "./dataset/template/CT_label_small.nii.gz"
        #print('source_path: ',source_path)
        target_path = os.path.join(self.source_root_dir, source_subfolder,"brain_small_norm.nii.gz")
        #target_label_path = "D:\\datasets\\20230423_pairs\\template\\CT_label_small.nii.gz"
        target_label_path = os.path.join(self.source_root_dir, source_subfolder, "label_small.nii.gz")
        #print('target_path: ',target_path)
        # 检查文件是否存在
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            raise FileNotFoundError(f"Source or target file not found for subfolder {source_subfolder}")

        syn_ct_image_sitk = sitk.ReadImage(source_path)
        target_image_sitk = sitk.ReadImage(target_path)
        syn_ct_label_sitk = sitk.ReadImage(syn_ct_label_path)
        target_label_sitk = sitk.ReadImage(target_label_path)

        syn_ct_image_np = sitk.GetArrayFromImage(syn_ct_image_sitk) #Converting sitk_metadata to image Array
        target_image_np = sitk.GetArrayFromImage(target_image_sitk)
        syn_ct_label_np = sitk.GetArrayFromImage(syn_ct_label_sitk)
        target_label_np = sitk.GetArrayFromImage(target_label_sitk)

        # 创建分类标签值到连续整数的映射
        label_mapping = {
            0.0: 0,
            1.0: 1,
            2.0: 2,
            3.0: 3,
            4.0: 4,
            5.0: 5,
            6.0: 6,
            7.0: 7,
            8.0: 8,
            9.0: 9,
            10.0: 10,
            11.0: 11,
            12.0: 12,
            13.0: 13,
            14.0: 14,
            15.0: 15,
            16.0: 16,
            17.0: 17,
            18.0: 18,
            19.0: 19,
            20.0: 20,
            21.0: 21,
            24.0: 22,
            25.0: 23,
            28.0: 24,
            29.0: 25,
            30.0: 26
        }

        # 使用映射替换标签数组中的值
        for old_label, new_label in label_mapping.items():
            syn_ct_label_np[syn_ct_label_np == old_label] = new_label
            target_label_np[target_label_np == old_label] = new_label
        # 现在，label1_array 包含了连续的整数值
        syn_ct_label_np = syn_ct_label_np.astype(int)
        target_label_np = target_label_np.astype(int)  
  
        syn_ct_image = torch.Tensor(syn_ct_image_np).unsqueeze(dim = 0)
        target_image = torch.Tensor(target_image_np).unsqueeze(dim = 0)
        syn_ct_label = torch.Tensor(syn_ct_label_np).unsqueeze(dim = 0)
        target_label = torch.Tensor(target_label_np).unsqueeze(dim = 0)
        
        source_data_dict = {'image': syn_ct_image, "label": syn_ct_label}
        target_data_dict = {'image':target_image, "label": target_label}
        # Apply transformation
        if self.transform2img:
            syn_ct_image = apply_transform(self.transform2img, syn_ct_image)
            target_image = apply_transform(self.transform2img, target_image)

        if self.transform2both:
            trans_source_data = apply_transform(self.transform2both, source_data_dict)
            trans_target_data = apply_transform(self.transform2both, target_data_dict)
            
            syn_ct_image = trans_source_data['image']
            syn_ct_label = trans_source_data['label']
            target_image = trans_target_data['image']
            target_label = trans_target_data['label']
            
        
        
        return syn_ct_image,target_image,syn_ct_label,target_label

class OASIS_Dataset(Dataset):
    def __init__(self, source_root_dir,transform2img=None,transform2both=None,is_Training=True,is_val=False):
        
        self.source_root_dir = source_root_dir
        #self.target_root_dir = target_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        
        if is_Training:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][:335]
        elif is_val:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][335:375]
        else:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][375:]
        #self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)]
        
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)]
    def __len__(self):
        return len(self.source_subfolders) 
        
    def __getitem__(self, idx):
        
        source_subfolder = self.source_subfolders[idx]
        #print(source_subfolder)
        #print(self.source_subfolders)
        
        # 构建源图像和目标图像的文件路径
        source_path = os.path.join(self.source_root_dir, source_subfolder, "aligned_norm.nii.gz")
        source_label_path = os.path.join(self.source_root_dir, source_subfolder, "aligned_seg35.nii.gz")
        #print('source_path: ',source_path)

        # 检查文件是否存在
        if not os.path.exists(source_path) :
            raise FileNotFoundError(f"Source file not found for subfolder {source_subfolder}")

        source_image_sitk = sitk.ReadImage(source_path)
        source_label_sitk = sitk.ReadImage(source_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk) #Converting sitk_metadata to image Array
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)

        # 现在，label1_array 包含了连续的整数值
        #source_label_np = source_label_np.astype(int)
        source_image = torch.Tensor(source_image_np).unsqueeze(dim = 0)

        source_label = torch.Tensor(source_label_np).unsqueeze(dim = 0)

        data_dict = {'image': source_image, "label": source_label}

        # Apply transformation
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)

        if self.transform2both:
            trans_data = apply_transform(self.transform2both, data_dict)
            source_image = trans_data['image']
            source_label = trans_data['label']
            
            
        return source_image,source_label

class Syn_OASIS_Dataset(Dataset):
    def __init__(self, source_root_dir,source_label_dir,transform2img=None,transform2both=None):
        
        self.source_root_dir = source_root_dir
        #self.target_root_dir = target_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        self.source_image_paths = []
        self.source_label_paths = []
        for i in range(1, 373):
            #source_path = os.path.join(source_root_dir, f"pred_image_OASIS_OAS1_{i:04}_MR1.nii.gz")
            source_path = os.path.join(source_root_dir, f"OASIS_OAS1_{i:04}_MR1/aligned_norm.nii.gz")
            source_label_path = os.path.join(source_label_dir, f"pred_label_OASIS_OAS1_{i:04}_MR1.nii.gz")
            if os.path.exists(source_path) and os.path.exists(source_label_path):
                self.source_image_paths.append(source_path)
                self.source_label_paths.append(source_label_path)
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][1:295]
        
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)]
    def __len__(self):
        return len(self.source_image_paths) 
        
    def __getitem__(self, idx):
                
        # 构建源图像和目标图像的文件路径
        source_path = self.source_image_paths[idx]
        source_label_path = self.source_label_paths[idx]
        #print('source_path: ',source_path)

        # 检查文件是否存在
        if not os.path.exists(source_path) :
            raise FileNotFoundError(f"Source file not found for source_path {source_path}")

        source_image_sitk = sitk.ReadImage(source_path)
        source_label_sitk = sitk.ReadImage(source_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk) #Converting sitk_metadata to image Array
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)

        # 现在，label1_array 包含了连续的整数值
        #source_label_np = source_label_np.astype(int)
        source_image = torch.Tensor(source_image_np).unsqueeze(dim = 0)

        source_label = torch.Tensor(source_label_np).unsqueeze(dim = 0)

        data_dict = {'image': source_image, "label": source_label}

        # Apply transformation
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)

        if self.transform2both:
            trans_data = apply_transform(self.transform2both, data_dict)
            source_image = trans_data['image']
            source_label = trans_data['label']
            
            
        return source_image,source_label        

class Syn_real_OASIS_Dataset(Dataset):
    def __init__(self, source_root_dir,target_root_dir,transform2img=None,transform2both=None):
        
        self.source_root_dir = source_root_dir
        self.target_root_dir = target_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        self.source_image_paths = []
        self.fusion_label_paths = []
        self.real_image_paths = []
        for i in range(1, 373):
            source_path = os.path.join(source_root_dir, f"pred_image_OASIS_OAS1_{i:04}_MR1.nii.gz")
            fusion_label_path = os.path.join(source_root_dir, f"pred_label_OASIS_OAS1_{i:04}_MR1.nii.gz")
            real_path = os.path.join(target_root_dir, f"OASIS_OAS1_{i:04}_MR1/aligned_norm.nii.gz")
            if os.path.exists(source_path) and os.path.exists(fusion_label_path) and os.path.exists(real_path):
                self.source_image_paths.append(source_path)
                self.fusion_label_paths.append(fusion_label_path)
                self.real_image_paths.append(real_path)
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][1:295]
        
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)]
    def __len__(self):
        return len(self.source_image_paths) 
        
    def __getitem__(self, idx):
                
        # 构建源图像和目标图像的文件路径
        source_path = self.source_image_paths[idx]
        fusion_label_path = self.fusion_label_paths[idx]
        real_path = self.real_image_paths[idx]
        #print('real_path: ',real_path)

        # 检查文件是否存在
        if not os.path.exists(source_path)  :
            raise FileNotFoundError(f"Source file not found for source_path {source_path}")

        source_image_sitk = sitk.ReadImage(source_path)
        source_label_sitk = sitk.ReadImage(fusion_label_path)
        real_image_sitk = sitk.ReadImage(real_path)
        
        source_image_np = sitk.GetArrayFromImage(source_image_sitk) #Converting sitk_metadata to image Array
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)
        real_image_np = sitk.GetArrayFromImage(real_image_sitk)
        # 现在，label1_array 包含了连续的整数值
        #source_label_np = source_label_np.astype(int)
        source_image = torch.Tensor(source_image_np).unsqueeze(dim = 0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim = 0)
        real_image = torch.Tensor(real_image_np).unsqueeze(dim = 0)
        
        data_dict = {'image': source_image, "label": source_label}

        # Apply transformation
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)

        if self.transform2both:
            trans_data = apply_transform(self.transform2both, data_dict)
            source_image = trans_data['image']
            source_label = trans_data['label']
            
            
        return source_image,source_label,real_image   
        
class SYN_CT_MRI_Dataset(Dataset):
    def __init__(self, syn_image_root_dir,real_img_root_dir,transform2img=None,transform2both=None,change_label=True):     
        self.syn_image_root_dir = syn_image_root_dir
        self.real_img_root_dir = real_img_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        self.change_label = change_label
        # 获取所有子文件夹的名称
        self.source_image_paths = []
        self.fusion_label_paths = []
        self.real_image_paths = []
        for i in range(1, 71):
            source_path = os.path.join(self.syn_image_root_dir, f"pred_image_BCH_MRI{i:03}.nii.gz")
            fusion_label_path = os.path.join(self.syn_image_root_dir, f"pred_label_BCH_MRI{i:03}.nii.gz")
            real_path = os.path.join(self.real_img_root_dir, f"BCH_MRI{i:03}/brain_small_norigid_norm.nii.gz")

            if os.path.exists(source_path) and os.path.exists(fusion_label_path) and os.path.exists(real_path):
                self.source_image_paths.append(source_path)
                self.fusion_label_paths.append(fusion_label_path)
                self.real_image_paths.append(real_path)
        
        
    def __len__(self):
        return len(self.source_image_paths) 
        
    def __getitem__(self, idx):
        
        # 构建源图像和目标图像的文件路径
        syn_ct_image_path = self.source_image_paths[idx]
        syn_ct_label_path = self.fusion_label_paths[idx]
        real_mri_image_path = self.real_image_paths[idx]
        #print("syn_ct_image_path",syn_ct_image_path)
        #print("real_mri_image_path",real_mri_image_path)
        
        '''
        print('syn_ct_image_path: ',syn_ct_image_path)
        print('syn_ct_label_path: ',syn_ct_label_path)
        print('real_mri_image_path: ',real_mri_image_path)
        '''

        syn_ct_image_sitk = sitk.ReadImage(syn_ct_image_path)
        syn_ct_label_sitk = sitk.ReadImage(syn_ct_label_path)
        real_mri_image_sitk = sitk.ReadImage(real_mri_image_path)
        
        syn_ct_image_np = sitk.GetArrayFromImage(syn_ct_image_sitk) #Converting sitk_metadata to image Array
        syn_ct_label_np = sitk.GetArrayFromImage(syn_ct_label_sitk)
        real_mri_image_np = sitk.GetArrayFromImage(real_mri_image_sitk)
        
        if self.change_label:
            # 创建分类标签值到连续整数的映射
            label_mapping = {
                0.0: 0,
                1.0: 1,
                2.0: 1,
                3.0: 1,
                4.0: 1,
                5.0: 1,
                6.0: 1,
                7.0: 1,
                8.0: 1,
                9.0: 2,
                10.0: 2,
                11.0: 2,
                12.0: 2,
                13.0: 2,
                14.0: 5,
                15.0: 5,
                16.0: 3,
                17.0: 3,
                18.0: 3,
                19.0: 3,
                20.0: 4,
                21.0: 4,
                22.0: 4,
                23.0: 4,
                24.0: 5,
                25.0: 5,
                26.0: 5
            }

            # 使用映射替换标签数组中的值
            for old_label, new_label in label_mapping.items():
                syn_ct_label_np[syn_ct_label_np == old_label] = new_label
    
        
        syn_ct_label_np = syn_ct_label_np.astype(int)  
  
        syn_ct_image = torch.Tensor(syn_ct_image_np).unsqueeze(dim = 0)
        syn_ct_label = torch.Tensor(syn_ct_label_np).unsqueeze(dim = 0)
        real_mri_image = torch.Tensor(real_mri_image_np).unsqueeze(dim = 0)
        
        source_data_dict = {'image_mri': real_mri_image,'image_ct': syn_ct_image, "label": syn_ct_label}
        # Apply transformation
        if self.transform2img:
            syn_ct_image = apply_transform(self.transform2img, syn_ct_image)
            real_mri_image = apply_transform(self.transform2img, real_mri_image)
        if self.transform2both:
            trans_source_data = apply_transform(self.transform2both, source_data_dict)
            
            syn_ct_image = trans_source_data['image_ct']
            syn_ct_label = trans_source_data['label']
            real_mri_image = trans_source_data['image_mri']
        
        return syn_ct_image,syn_ct_label,real_mri_image

class Seg_CT_MRI_Dataset(Dataset):
    def __init__(self, source_root_dir,transform2img=None,transform2both=None,is_Training=True,is_val=False):
        
        self.source_root_dir = source_root_dir
        #self.target_root_dir = target_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        if is_Training:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][:70]
        elif is_val:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][70:88]
        else:
            self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)][88:]
        #self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)]
    
    def __len__(self):
        return len(self.source_subfolders) 
        
    def __getitem__(self, idx):
        
        source_subfolder = self.source_subfolders[idx]
        #print(source_subfolder)
        #print(self.source_subfolders)
        
        # 构建源图像和目标图像的文件路径
        source_path = os.path.join(self.source_root_dir, source_subfolder, "brain_small_norm.nii.gz")
        source_label_path = os.path.join(self.source_root_dir, source_subfolder, "label_small.nii.gz")
        #print('real ct image path: ',source_path)

        # 检查文件是否存在
        if not os.path.exists(source_path) :
            raise FileNotFoundError(f"Source file not found for subfolder {source_subfolder}")

        source_image_sitk = sitk.ReadImage(source_path)
        source_label_sitk = sitk.ReadImage(source_label_path)

        source_image_np = sitk.GetArrayFromImage(source_image_sitk) #Converting sitk_metadata to image Array
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)
        '''
        # 创建分类标签值到连续整数的映射
        label_mapping = {
            0.0: 0,
            1.0: 1,
            2.0: 2,
            3.0: 3,
            4.0: 4,
            5.0: 5,
            6.0: 6,
            7.0: 7,
            8.0: 8,
            9.0: 9,
            10.0: 10,
            11.0: 11,
            12.0: 12,
            13.0: 13,
            14.0: 14,
            15.0: 15,
            16.0: 16,
            17.0: 17,
            18.0: 18,
            19.0: 19,
            20.0: 20,
            21.0: 21,
            24.0: 22,
            25.0: 23,
            28.0: 24,
            29.0: 25,
            30.0: 26
        }
        '''
        label_mapping = {
            0.0: 0,
            1.0: 1,
            2.0: 2,
            3.0: 1,
            4.0: 2,
            5.0: 1,
            6.0: 2,
            7.0: 1,
            8.0: 2,
            9.0: 4,
            10.0: 3,
            11.0: 4,
            12.0: 3,
            13.0: 0,
            14.0: 1,
            15.0: 2,
            16.0: 0,
            17.0: 0,
            18.0: 1,
            19.0: 2,
            20.0: 6,
            21.0: 5,
            24.0: 6,
            25.0: 5,
            28.0: 0,
            29.0: 1,
            30.0: 2
        }
        
        # 使用映射替换标签数组中的值
        for old_label, new_label in label_mapping.items():
            source_label_np[source_label_np == old_label] = new_label

        # 现在，label1_array 包含了连续的整数值
        source_label_np = source_label_np.astype(int)


        source_image = torch.Tensor(source_image_np).unsqueeze(dim = 0)

        source_label = torch.Tensor(source_label_np).unsqueeze(dim = 0)

        data_dict = {'image': source_image, "label": source_label}

        # Apply transformation
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)

        if self.transform2both:
            trans_data = apply_transform(self.transform2both, data_dict)
            source_image = trans_data['image']
            source_label = trans_data['label']
            
            
        return source_image,source_label

class SYN_CT2CT_Dataset(Dataset):
    def __init__(self, syn_image_root_dir,transform2img=None,transform2both=None):     
        self.syn_image_root_dir = syn_image_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        # 获取所有子文件夹的名称
        self.source_image_paths = []
        self.source_label_paths = []
        for i in range(1, 71):
            source_path = os.path.join(self.syn_image_root_dir, f"pred_image_BCH_MRI{i:03}.nii.gz")
            source_label_path = os.path.join(self.syn_image_root_dir, f"pred_label_BCH_MRI{i:03}.nii.gz")
     
            if os.path.exists(source_path) and os.path.exists(source_label_path):
                self.source_image_paths.append(source_path)
                self.source_label_paths.append(source_label_path)

        
        
    def __len__(self):
        return len(self.source_image_paths) 
        
    def __getitem__(self, idx):
        
        # 构建源图像和目标图像的文件路径
        syn_ct_image_path = self.source_image_paths[idx]
        syn_ct_label_path = self.source_label_paths[idx]
        #print("syn_ct_image_path",syn_ct_image_path)
        #print("real_mri_image_path",real_mri_image_path)
        
        '''
        print('syn_ct_image_path: ',syn_ct_image_path)
        print('syn_ct_label_path: ',syn_ct_label_path)
        '''

        syn_ct_image_sitk = sitk.ReadImage(syn_ct_image_path)
        syn_ct_label_sitk = sitk.ReadImage(syn_ct_label_path)
        
        syn_ct_image_np = sitk.GetArrayFromImage(syn_ct_image_sitk) #Converting sitk_metadata to image Array
        syn_ct_label_np = sitk.GetArrayFromImage(syn_ct_label_sitk)
        
        syn_ct_label_np = syn_ct_label_np.astype(int)  
  
        syn_ct_image = torch.Tensor(syn_ct_image_np).unsqueeze(dim = 0)
        syn_ct_label = torch.Tensor(syn_ct_label_np).unsqueeze(dim = 0)
        
        source_data_dict = {'image': syn_ct_image, "label": syn_ct_label}
        # Apply transformation
        if self.transform2img:
            syn_ct_image = apply_transform(self.transform2img, syn_ct_image)
        if self.transform2both:
            trans_source_data = apply_transform(self.transform2both, source_data_dict)
            
            syn_ct_image = trans_source_data['image']
            syn_ct_label = trans_source_data['label']

        
        return syn_ct_image,syn_ct_label

class Seg_IXI_Dataset(Dataset):
    def __init__(self, source_root_dir,transform2img=None,transform2both=None,is_Training=True,is_val=False):
        
        self.source_root_dir = source_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        
        if is_Training:
            self.source_files = [filename for filename in os.listdir(source_root_dir)]
        elif is_val:
            self.source_files = [filename for filename in os.listdir(source_root_dir)]
        else:
            self.source_files = [filename for filename in os.listdir(source_root_dir)]
        #self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)]
        
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)]
    def __len__(self):
        return len(self.source_files) 
        
    def __getitem__(self, idx):
        
        source_filename = self.source_files[idx]
        #print(source_subfolder)
        #print(self.source_subfolders)
        
        # 构建源图像和目标图像的文件路径
        source_path = os.path.join(self.source_root_dir,source_filename)
        source_image_np,source_label_np = pkload(source_path) 
        
        source_image_np = np.transpose(source_image_np, (2,1,0))
        source_label_np = np.transpose(source_label_np, (2,1,0))

        
        source_image_np = np.ascontiguousarray(source_image_np)
        source_label_np = np.ascontiguousarray(source_label_np)
        
        #[0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63]
        label_mapping = {
            0 : 0,
            5 : 0,
            26: 0,
            30: 0,
            44: 0,
            58: 0,
            62: 0,
            72: 0,
            77: 0,
            80: 0,
            85: 0,
            251:0,
            252:0,
            253:0,
            254:0,
            255:0,
            2: 1,
            3: 2,
            4: 3,
            7: 4,
            8: 5,
            10: 6,
            11: 7,
            12: 8,
            13: 9,
            14: 10,
            15: 11,
            16: 12,
            17: 13,
            18: 14,
            24: 15,
            28: 16,
            31: 17,
            41: 18,
            42: 19,
            43: 20,
            46: 21,
            47: 22,
            49: 23,
            50: 24,
            51: 25,
            52: 26,
            53: 27,
            54: 28,
            60: 29,
            63: 30
        }

        # 使用映射替换标签数组中的值
        for old_label, new_label in label_mapping.items():
            source_label_np[source_label_np == old_label] = new_label

        # 现在，label1_array 包含了连续的整数值
        source_label_np = source_label_np.astype(int) 

        # 现在，label1_array 包含了连续的整数值
        #source_label_np = source_label_np.astype(int)
        source_image = torch.Tensor(source_image_np).unsqueeze(dim = 0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim = 0)
        
        source_data_dict = {'image': source_image, "label": source_label}
  
        # Apply transformation
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
        if self.transform2both:
            source_trans_data = apply_transform(self.transform2both, source_data_dict)
            source_image = source_trans_data['image']
            source_label = source_trans_data['label']

            
        return source_image,source_label
    
class Syn_real_IXI_Dataset(Dataset):
    def __init__(self, syn_root_dir,real_root_dir,transform2img=None,transform2both=None):

        self.syn_root_dir = syn_root_dir
        self.real_root_dir = real_root_dir
        self.transform2img = transform2img
        self.transform2both = transform2both
        #self.is_Training = is_Training
        # 获取所有子文件夹的名称
        
        self.source_image_paths = []
        self.fusion_label_paths = []
        self.real_image_paths = []
        for i in range(0, 577):
            source_path = os.path.join(syn_root_dir, f"pred_image_subject_{i}.nii.gz")
            fusion_label_path = os.path.join(syn_root_dir, f"pred_label_subject_{i}.nii.gz")
            real_path = os.path.join(real_root_dir, f"subject_{i}.pkl")

            if os.path.exists(source_path) and os.path.exists(fusion_label_path) and os.path.exists(real_path):
                self.source_image_paths.append(source_path)
                self.fusion_label_paths.append(fusion_label_path)
                self.real_image_paths.append(real_path)
         
        #self.target_subfolders = [subfolder for subfolder in os.listdir(target_root_dir)]
        
        #self.source_subfolders = [subfolder for subfolder in os.listdir(source_root_dir)]
    def __len__(self):
        return len(self.source_image_paths) 
        
    def __getitem__(self, idx):
        
        # 构建源图像和目标图像的文件路径
        source_path = self.source_image_paths[idx]
        fusion_label_path = self.fusion_label_paths[idx]
        real_path = self.real_image_paths[idx]
        #print('real_path: ',real_path)


        source_image_sitk = sitk.ReadImage(source_path)
        source_label_sitk = sitk.ReadImage(fusion_label_path)
        
        source_image_np = sitk.GetArrayFromImage(source_image_sitk) #Converting sitk_metadata to image Array
        source_label_np = sitk.GetArrayFromImage(source_label_sitk)
        real_image_np,_ = pkload(real_path) 
        real_image_np = np.transpose(real_image_np, (2,1,0))
        #real_label_np = np.transpose(real_label_np, (2,1,0))
        real_image_np = np.ascontiguousarray(real_image_np)    
        #real_label_np = np.ascontiguousarray(real_label_np)    

        # 现在，label1_array 包含了连续的整数值
        source_label_np = source_label_np.astype(int)
        #real_label_np = real_label_np.astype(int)
        # 现在，label1_array 包含了连续的整数值
        #source_label_np = source_label_np.astype(int)
        source_image = torch.Tensor(source_image_np).unsqueeze(dim = 0)
        source_label = torch.Tensor(source_label_np).unsqueeze(dim = 0)
        real_image = torch.Tensor(real_image_np).unsqueeze(dim = 0)
        #real_label = torch.Tensor(real_label_np).unsqueeze(dim = 0)
        source_data_dict = {'image': source_image, "label": source_label}
  
        # Apply transformation
        if self.transform2img:
            source_image = apply_transform(self.transform2img, source_image)
        if self.transform2both:
            source_trans_data = apply_transform(self.transform2both, source_data_dict)
            source_image = source_trans_data['image']
            source_label = source_trans_data['label']

            
        return source_image,source_label,real_image