# Subgraph-Aware Training for Knowledge Graph Completion(SATKGC) 

<p align="center">
<img width="1000" alt="1" src="https://github.com/meaningful96/SATKGC/assets/111734605/3ea3f187-7507-426e-8a9c-e4f9d9ce247c">
</p>

## Requirements
- python ≥ 3.9
- torch == 1.13.1
- transformer ≥ 4.33.1
- networkx == 3.2.1
```bash
# Make a virtual space
conda create -n SATKGC python=3.9 -y
conda activate SATKGC

# Install core packages
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformer==4.33.1
pip install networkx==3.2.1
```



## Step 1) Select the one of the SaaM methods (BRWR, MCMC)
```bash
# BRWR_IP: Biased Random Walk with Restart - Inversely Proportional
cd BRWR_IP

# MCMC: Markov Chain Monte-Carlo
cd MCMC
```

## Step 2) Data Preprocessing

### BRWR
#### Make `json` file from raw data
```bash
## WN18RR, FB15k-237, NELL-995
bash scripts/preprocess.sh WN18RR
bash scripts/preprocess.sh FB15k237
python3 preprocess_nell.py NELL-995

## Wikidata5M needs to download at first
bash ./scripts/download_wikidata5m.sh

## Make json file
bash scripts/preprocess.sh wiki5m_trans
bash scripts/preprocess.sh wiki5m_ind
```

#### Subgraph Sampling using BRWR
- Select the number of CPU cores.
- The more CPU cores, the faster the sampling
- `--k-step`
    - It is the maximum BRWR path length
    - The reciprocal of half of `--k-step` is the restart probability. Therefore, it is extremely rare for the process to reach the maximum step.
- `--n-iter`
    - This is the number of times BRWR is repeated. When `--k-step` is considered as a single path, a total of `--n-iter` paths are used as a subgraph.
- run `LKG_randomwalk.py`
  
```bash
python3 LKG_randomwalk.py \
--base-dir (SATKGC Path)/data \
--k-step 50 \
--n-iter  300 \
--dataset wiki5m_ind \
--distribution antithetical \
--phase 1 \
--subgraph-size 512 \
--mode train

# dataset: WN18RR, FB15k237, NELL-995, wiki5m_ind, wiki5m_trans
# mode: train, valid
```


### MCMC
#### Make `json` file from raw data
```bash
## WN18RR, FB15k-237
bash scripts/preprocess.sh WN18RR
bash scripts/preprocess.sh FB15k237

## Wikidata5M needs to download at first
bash ./scripts/download_wikidata5m.sh

## Make json file
bash scripts/preprocess.sh wiki5m_trans
bash scripts/preprocess.sh wiki5m_ind
```

## Step 3) Training(BRWR, MCMC)
#### WN18RR
```bash
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

#### FB15k237
```bash
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh
```

#### NELL-995
```bash
OUTPUT_DIR=./checkpoint/nell995/ bash scripts/train_nell.sh
```

#### Wikidata5M
```bash
# Inductive
OUTPUT_DIR=./checkpoint/wiki5m_ind/ bash scripts/train_wiki.sh wiki5m_ind

# Transductive
OUTPUT_DIR=./checkpoint/wiki5m_trans/ bash scripts/train_wiki.sh wiki5m_trans
```



## Step 4) Inference
#### WN18RR
```bash
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```

#### FB15k237
```bash
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
```

#### NELL-995
```bash
bash scripts/eval.sh ./checkpoint/nell995/model_last.mdl NELL-995
```

#### Wikidata5M
```bash
# Inductive
bash scripts/eval.sh ./checkpoint/wiki5m_ind/model_last.mdl wiki5m_ind

# Transductive
bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans/model_last.mdl
```

# Citation
```bash


```

# Acknowledgement
This code is based on [SimKGC](https://arxiv.org/abs/2203.02167).  


