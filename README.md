Here’s a complete, well-structured `README.md` tailored for your project, assuming it's located in a GitHub repo and follows best practices:

---

```markdown
# Knowledge Distillation for Neural Collaborative Filtering (NeuMF)

This project explores the effectiveness of **Knowledge Distillation** techniques in reducing the parameter size of the [NeuMF](https://arxiv.org/abs/1708.05031) model without significant loss in recommendation performance. It is based on the **MovieLens 100K** dataset and was completed as part of the university course **"Deep Learning and Its Applications"**.

## 🔍 Objectives

- Reproduce NeuMF results on a different dataset (MovieLens 100K).
- Investigate the impact of architecture choices (e.g. number of MLP layers, embedding size).
- Compare NeuMF to NMF in terms of HR@10, NDCG@10, and model size.
- Implement and evaluate 3 different **knowledge distillation** strategies:
  - Response-based distillation
  - Feature-based distillation
  - Relation-based distillation

## 📁 Repository Structure

```

.
├── ex\_12.py                 # Main training/evaluation script for student distillation
├── Dataset.py               # Dataset loader and preprocessor
├── evaluate.py              # Evaluation metrics (HR\@K, NDCG\@K)
├── NeuMF.py                 # NeuMF model definition
├── MLP.py                   # Standalone MLP (student model)
├── Pretrain/                # Folder with pre-trained teacher weights (.npy)
├── logs\*/                   # Output logs from distillation experiments
├── script\_12.sh             # Shell script to automate 10-run experiments
└── Data/
└── ml-100k/             # MovieLens 100K data files

````

## ⚙️ Installation

Create a conda environment (recommended):
```bash
conda create -n ncf python=3.6 tensorflow=1.14 numpy pandas
conda activate ncf
````

Install additional dependencies:

```bash
pip install scikit-learn matplotlib
```


## 📌 Notes

* All experiments are repeated **10 times**, and results are reported as **mean ± std**.
* Negative sampling (`--num_neg`) and architecture settings are configurable.
* Pretraining is optional; this project focuses on *training from scratch* + distillation.

## 📚 References

* Xiangnan He et al. "[Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)", WWW 2017.
* Hinton et al. "[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)", NIPS 2015.
* Knowledge Distillation Benchmark: [https://arxiv.org/abs/2006.05525](https://arxiv.org/abs/2006.05525)

## 👤 Authors
* **\Kleidonaris Dimitris** – MSc Student, Department of Electrical & Computer Engineering, University of Thessaly
* **\Aggelos Tzikas** – MSc Student, Department of Electrical & Computer Engineering, University of Thessaly

## 📅 Submission

