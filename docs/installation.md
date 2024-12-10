
# 🏗️️ Installation





## 1. Create conda environment

```bash
conda create -n vipocc python=3.10 -y
conda activate vipocc
```

## 2. Install PyTorch

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```