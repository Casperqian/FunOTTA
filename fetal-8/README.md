# Cross-Domain Image Classification on the Fetal-8 Ultrasound Dataset

## Overview

This experiment evaluates the cross-modality generalizability the **FunOTTA** by applying it to a maternal-fetal ultrasound classification task using the Fetal-8 dataset.
The Fetal-8 dataset represents a different modality from our primary benchmark (fundus images), thus serving as a robust testbed for verifying the adaptability of our model across imaging modalities.

## 📂 Dataset Description

The dataset comprises 10,850 ultrasound images acquired from two distinct machine vendors: ALOKA (A) and Voluson (V). 
It includes eight types of anatomical planes:  brain (3), abdomen (1), femur (1), thorax (1), maternal cervix (1), and others (1)—forming a multi-class classification task.

<p align="center">
  <img src="https://github.com/Casperqian/FunOTTA/blob/main/img/fetal-8_distribution.png" alt="Label Distribution by Vendor" width="700"/>
</p>

## ⚙️ Setup
We simulate domain shift by performing **cross-vendor validation**:
- **Train on ALOKA (A)** ➝ **Test on Voluson (V)**
- **Train on Voluson (V)** ➝ **Test on ALOKA (A)**

## 📈 Results
The results on the Fetal-8 dataset are shown below:
<p align="center">
  <img src="https://github.com/Casperqian/FunOTTA/blob/main/img/fetal-8_results.png" alt="Label Distribution by Vendor" width="400"/>
</p>

## 📚 Citation
### 🔹 Fetal-8 Dataset

> Xavier P. Burgos-Artizzu, David Coronado-Gutiérrez, Brenda Valenzuela-Alcaraz, Elisenda Bonet-Carne, Elisenda Eixarch, Fatima Crispi, Eduard Gratacós.  
> **Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes.**  
> *Scientific Reports*, 10, 10200 (2020).  
> https://doi.org/10.1038/s41598-020-66747-4

```bibtex
@article{burgos2020evaluation,
  title={Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes},
  author={Burgos-Artizzu, Xavier P and Coronado-Guti{\'e}rrez, David and Valenzuela-Alcaraz, Brenda and Bonet-Carne, Elisenda and Eixarch, Elisenda and Crispi, Fatima and Gratac{\'o}s, Eduard},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={10200},
  year={2020},
  publisher={Nature Publishing Group UK London}
}





