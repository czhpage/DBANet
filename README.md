# DBANet: Dual Boundary Awareness With Confidence-Guided Pseudo Labeling for Medical Image Segmentation 

This is the official PyTorch implementation of our paper:

> **DBANet: Dual Boundary Awareness With Confidence-Guided Pseudo Labeling for Medical Image Segmentation**  
> IEEE Journal of Biomedical and Health Informatics (JBHI), 2025  
> [📄 Paper](https://ieeexplore.ieee.org/document/11104802) | [🌐 Project Page](https://github.com/czhpage/DBANet)

---

## 🔥 Highlights
- 🏆 State-of-the-art performance on **Synapse**, **WORD** and **AMOS** datasets.  
- ⚡ Efficient training and inference.  
- 🔧 Easy to adapt to your own medical image segmentation tasks.
- 📚 Clear comparisons with existing semi-supervised methods, making it easy to understand and learn, especially for newcomers.

---

## 📦 Installation
```bash
git clone https://github.com/czhpage/DBANet.git
cd DBANet
conda create -n dbanet python=3.8.2 -y
conda activate dbanet
pip install -r requirements.txt
```

---

## 🛠 Environment

Before running the code, set the PYTHONPATH to the project root:

```bash
export PYTHONPATH=$(pwd)/code:$PYTHONPATH
```
---

## 📂 Dataset Preparation  

### Dataset Download
The raw datasets can be obtained from the following sources:  
- **Synapse**: (https://www.synapse.org/Synapse:syn3193805/wiki/89480)  
- **AMOS**: (https://amos22.grand-challenge.org/)  
- **WORD**: (https://github.com/HiLab-git/WORD?tab=readme-ov-file)  

### Preprocessing
📌 Please follow the preprocessing steps described in [DHC](https://github.com/xmed-lab/DHC) **Data Preparation** section.  

---

### Preprocessed Datasets (Optional, More Convenient)
⚡ Alternatively, you may directly download the preprocessed datasets from the provided links (Link 1, Link 2, Link 3) corresponding to **Synapse**, **WORD**, and **AMOS**.  

---

### 🗂 Folder Structure
The processed dataset folders are organized as follows:  
```bash
./synapse_data/
├── npy
│ ├── <id>_image.npy
│ ├── <id>_label.npy
├── processed
│ ├── <id>image.nii.gz
│ ├── <id>label.nii.gz
├── splits
│ ├── labeled_20p.txt
│ ├── unlabeled_20p.txt
│ ├── train.txt
│ ├── eval.txt
│ ├── test.txt
│ ├── ...

./word_data/
├── npy
│ ├── <id>_image.npy
│ ├── <id>_label.npy
├── splits
│ ├── labeled_20p.txt
│ ├── unlabeled_20p.txt
│ ├── train.txt
│ ├── eval.txt
│ ├── test.txt
│ ├── ...

./amos_data/
├── npy
│ ├── <id>_image.npy
│ ├── <id>_label.npy
├── splits
│ ├── labeled_20p.txt
│ ├── unlabeled_20p.txt
│ ├── train.txt
│ ├── eval.txt
│ ├── test.txt
│ ├── ...
```

---

## 🚀 Training
Run semi-supervised training, evaluating and testing with:
```bash
bash train3times_seeds_10p.sh -c 0 -t synapse -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_20p.sh -c 0 -t synapse -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_40p.sh -c 0 -t synapse -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_2p.sh -c 0 -t word -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_5p.sh -c 0 -t word -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_10p.sh -c 0 -t word -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_2p.sh -c 0 -t amos -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_5p.sh -c 0 -t amos -m DBANet -e '' -l 3e-2 -w 0.1
bash train3times_seeds_10p.sh -c 0 -t amos -m DBANet -e '' -l 3e-2 -w 0.1

```

---
## 📊 Results

Here we show qualitative segmentation results under a 40% labeled / 60% unlabeled Synapse training set :

![Qualitative Results](images/Visualization.png)

---

## 📖 Citation

If you find this work useful, please cite:
```
@ARTICLE{11104802,
  author={Chen, Zhonghua and Cao, Haitao and Kettunen, Lauri and Wang, Hongkai},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={DBANet: Dual Boundary Awareness With Confidence-Guided Pseudo Labeling for Medical Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/JBHI.2025.3592873}}
```

## 🤝 Acknowledgements

We thank the authors of [DHC](https://github.com/xmed-lab/DHC) for providing their code and benchmarks, which facilitated our comparative experiments. Our implementation builds upon their work, while introducing novel components and training strategies specific to DBANet. We also acknowledge the author of a publicly available [GitHub repository](https://github.com/yiskw713/boundary_loss_for_remote_sensing), which offered helpful insights during our development.
