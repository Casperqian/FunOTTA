# Test-Time Adaptation for Fundus Image Classification

This is the official implementation of:

> **FunOTTA: On-the-Fly Adaptation on Cross-Domain Fundus Image via Stable Test-time Training**  *(Under Review)*

<p align="center">
  <img src="https://github.com/Casperqian/FunOTTA/blob/main/img/Overview_v2.png" alt="Framework Diagram" width="90%" />
</p>

## 🚀 Get Started

Clone the repository:

```bash
git clone https://github.com/Casperqian/FunOTTA.git
cd FunOTTA
````

## 🔧 Training & Test-Time Adaptation

### Step 1: Train the source model

```bash
bash train.sh
```

### Step 2: Apply test-time adaptation

```bash
bash adaptation.sh
```

Test-time adaptation methods are modular and can be customized in `./domainbed/adapt_algorithms`.

## 🧪 Application to Other Modalities

To validate the cross-modality generalizability of FunOTTA, we applied it to ultrasound images using the Fetal-8 dataset. This dataset features a different imaging modality with domain shift between vendors.

* 📁 Folder: [`./fetal-8/`](./fetal-8/)
* 📄 Dataset: Fetal-8 maternal-fetal ultrasound (ALOKA vs. Voluson)
* ✅ Task: Cross-vendor classification across 8 anatomical classes

See [`./fetal-8/README.md`](./fetal-8/README.md) for details.
