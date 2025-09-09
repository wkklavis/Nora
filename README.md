# Noise-Robust Tuning of SAM for Domain Generalized Ultrasound Image Segmentation
This is the official code of our MICCAI 2025 paper Nora 🥳

<div align=center>
	<img src="figures/Pipeline.png" width=75%/>
</div>

## Requirement
```
conda create -n Nora python=3.7.16
conda activate Nora
pip install -r requirements.txt
```

## Data Preparation

- **BUS**: [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [DatasetB](https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php), [STU](https://github.com/xbhlk/STU-Hospital)  
- **Thyroid**: [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [DDTI processed](https://github.com/ynatovich/mlhc/tree/main/2_preprocessed_data)  
- **MYO**: [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html), [HMC-QU](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset)  

> Process MYO data with: `dataset/MYO/CAMUS_process.py`, `dataset/MYO/HMC_QU_process.py`

---

## Directory Structure

```plaintext
├── BUS
│   ├── BUSI
│   │   ├── images
│   │   └── masks
│   ├── DatasetB
│   │   ├── BUS
│   │   │   ├── image
│   │   │   └── label
│   └── STU
│       └── BUSI
│           ├── img
│           └── label
│
├── Thyroid
│   ├── DDTI dataset
│   │   └── DDTI
│   │       └── 2_preprocessed_data
│   │           └── stage2
│   │               ├── p_image
│   │               └── p_mask
│   ├── tn3k
│   │   ├── trainval-image
│   │   └── trainval-mask
│
├── MYO
│   ├── CAMUS3K
│   │   ├── images
│   │   └── labels
│   └── HMC-QU
│       ├── images
│       └── labels
```

Please download the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) 
(provided by the original repository of SAM) and put it in the ./pretrained folder. 
What's more, we also provide well-trained models at [Release](https://github.com/wkklavis/Nora/releases/tag/v.1.0.0). Please put it in the ./snapshot folder for evaluation. 

## Training
**BUS** (BUSI → DatasetB, STU)  
```
CUDA_VISIBLE_DEVICES=0 python train.py --source_root_path busi_dataset_path --target_root_path1 datasetb_dataset_path --target_root_path2 stu_dataset_path 
```

**Thyroid** (TN3K->DDTI) 
```
CUDA_VISIBLE_DEVICES=0 python train_thyroid.py --source_root_path tn3k_dataset_path --target_root_path ddti_dataset_path 
```

**MYO** (CAMUS->HMC-QU)
```
CUDA_VISIBLE_DEVICES=0 python train_myo.py --source_root_path camus3k_dataset_path --target_root_path hmcqu_dataset_path 
```
