# DBANet: Dual Boundary Awareness With Confidence-Guided Pseudo Labeling for Medical Image Segmentation 

This is the official PyTorch implementation of our paper:

> **DBANet: Dual Boundary Awareness With Confidence-Guided Pseudo Labeling for Medical Image Segmentation**  
> IEEE Journal of Biomedical and Health Informatics (JBHI), 2025  
> [📄 Paper](https://ieeexplore.ieee.org/document/11104802) | [🌐 Project Page](https://github.com/czhpage/DBANet)

---

## 🔥 Highlights
- 🏆 State-of-the-art performance on **Synapse** and **WORD** datasets.  
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
## 📊 Results

Here we show qualitative segmentation results:

![Qualitative Results](assets/results.png)

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
