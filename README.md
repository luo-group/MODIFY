# MODIFY
ML-optimized library design with improved fitness and diversity for protein engineering

MODIFY (ML-optimized library design with improved fitness and diversity) is a machine learning algorithm for cold-start library design in protein engineering. Given a set of specified residues in a parent protein, MODIFY designs a high-fitness, high-diversity starting library of combinatorial variants.

The repository of MODIFY contained the implementation of the 3 components of MODIFY: ***1. zero-shot protein fitness prediction***, ***2. pareto optimization of fitness and diversity for library design***, and ***3. structure-based filtering***. For a high-level and holistic understanding our project, please refer to section [***2. pareto optimization of fitness and diversity for library design***](#2-pareto-optimization-of-fitness-and-diversity-for-library-design) first and check out the jupyter notebook (`notebooks/demo_GB1.ipynb`)

## Table of contents
- [MODIFY](#modify)
  - [Table of contents](#table-of-contents)
  - [Install dependencies](#install-dependencies)
  - [1. Zero-shot protein fitness prediction](#1-zero-shot-protein-fitness-prediction)
  - [2. Pareto optimization of fitness and diversity for library design](#2-pareto-optimization-of-fitness-and-diversity-for-library-design)
  - [3. Structure-based filtering](#3-structure-based-filtering)
  - [Contact](#contact)


## Install dependencies
First, clone the repo from github and  `data.zip`.
```bash
# Clone the github repo
git clone https://github.com/luo-group/MODIFY.git
cd MODIFY

# Download the data file and unzip it
wget -nv -O data.zip https://www.dropbox.com/scl/fi/c0ypz3d0vzfva8vbdb7f4/data.zip?rlkey=qg0f3v2dmhongcc32hpc3zbxt&dl=1
unzip data.zip

# Create folder for results
mkdir results
```

Then, create a conda environment for MODIFY.
```bash
conda create -n modify python=3.9
conda activate modify
```

Install dependencies for [***2. pareto optimization of fitness and diversity for library design***](#2-pareto-optimization-of-fitness-and-diversity-for-library-design). We use pytorch 2.0.0 and torchvision 0.15.0, which can be installed with the proper version for your CUDA or CPU following the instructions on the offical website of PyTorch (https://pytorch.org/). 
```bash
pip install -r requirements.txt

# install tensorboard
conda install -c conda-forge tensorboard
```

(Optional) If you want to perform [***1. zero-shot protein fitness prediction***](#1-zero-shot-protein-fitness-prediction) using this repository, install the following dependencies. As the computation will take a considerable amount of time, we have provided pre-calculated zero-shot predictions for the demonstration.
<!-- If you have already performed zero-shot protein fitness prediction using your own tool or the same tools as ours, you can upload the fitness predictions and organize the predictions in the same format as ours (as described in the following section). -->

```bash
# ESM-1v and ESM-2

pip install fair-esm

# EVcouplings
pip install evcouplings
pip3 install -U scikit-learn scipy
pip uninstall numpy==1.26.0
pip install numpy==1.23.1

# EVE
# Follow the instructions from https://github.com/OATML-Markslab/EVE.

# MSA Transformer
conda install -c conda-forge -c bioconda hhsuite
```

(Optional) If you want to perform [***3. structure-based filtering***](#3-structure-based-filtering) using this repository, install the following dependencies. As the computation will take a considerable amount of time, we have provided pre-calculated FoldX ddG and ESMFold pLDDT for the demonstration.
```bash
# ESMFold
conda env create -f environment.yml
# or 
conda create -n esmfold python==3.7
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
pip install biotite
# Note: there might be certain issues during installation, and you can refer to the public repo of ESMFold to solve the issues. This is because the installation here in our repo directly follows the public repo of ESMFold

# FoldX
# Download from https://foldxsuite.crg.eu/
```


## 1. Zero-shot protein fitness prediction

MODIFY integrates four pre-trained unsupervised ML models (ESM, EVmutation, EVE, and MSA Transformer) for zero-shot protein fitness prediction. As the computation for calculating the predictions for each method is time-consuming and resource-intensive, we provided our scripts for computing and also provide pre-calculated predictions. You can skip this section and directly jump to the next section [2. Pareto optimization of fitness and diversity for library design](#2-pareto-optimization-of-fitness-and-diversity-for-library-design).

For ESM-1v and ESM-2, we downloaded the scripts from the public repo of ESM (https://github.com/facebookresearch/esm#esmfold) and modified the scripts to adapt to multiprocessing. Please use `torch.hub.set_dir('')` to set a directory for ESM pre-trained models downloading, as this would usually take up a large amount of space. For detailed zero-shot fitness prediction implementation, please refer to `notebooks/zero.ipynb`.

For EVmutation, we generated the MSA from the server of EVcouplings (https://v2.evcouplings.org/) and downloaded the EVmutation model from the results page. For zero-shot fitness prediction implementation, please refer to `notebooks/zero.ipynb`.

For EVE, we used the same MSA as EVmutation, and used the code from the public repo (https://github.com/OATML-Markslab/EVE). We modified the scripts to enable multiprocessing inference. For zero-shot fitness prediction implementation, please refer to `notebooks/zero.ipynb`.

For MSA Transformer, we downloaded the scripts from the public repo of ESM (https://github.com/facebookresearch/esm#esmfold) and modified the scripts to adapt to multiprocessing. We used hhfilter to subsample the MSA. Please use `torch.hub.set_dir('')` to set a directory for ESM pre-trained models downloading, as this would usually take up a large amount of space. For detailed zero-shot fitness prediction implementation, please refer to `notebooks/zero.ipynb`


## 2. Pareto optimization of fitness and diversity for library design

Here in our repository, we used the pre-calculated zero-shot protein fitness predictions of GB1 variants `data/GB1/GB1_zero.csv` as the demo and showed how to perform the pareto optimization.  

First, we implement the default setting of MODIFY, which assigns equal residue-level diveristy weights to each target residue. The search space of $\lambda$ (equivalent to $\lambda\alpha_i$ in the paper) is from 0 to 2, with increments of 0.01. The designed libraries as parameterized by $\lambda$ are stored at `results/GB1`. The log files for tensorboard are stored in `log/` and you can use `tensorboard` to check the log for stochastic gradient ascent.

```bash
python scripts/run_modify.py \
        --protein GB1 \
        --offset 1 \
        --positions 39,40,41,54 \
        --fitness_col modify_fitness \
        --parallel \
        --num_proc 60 \
        --seed 29
```

Or, if you would like to mask certain AAs (e.g., 39L and 41G).

```bash
python scripts/run_modify.py \
        --protein GB1 \
        --offset 1 \
        --positions 39,40,41,54 \
        --masked_AAs 39L,41G \
        --fitness_col modify_fitness \
        --parallel \
        --num_proc 60 \
        --seed 29
```

For informed setting, we followed the example we showed in our paper, which adjusts the diversity at site 40. The argument for adjusting site-wise diversity is formated as '{site}-{lambda}'. Here, $\lambda$ is equivalent to $\lambda\alpha_i$ in the paper.
```bash
python scripts/run_modify.py \
        --protein GB1 \
        --offset 1 \
        --positions 39,40,41,54 \
        --fitness_col modify_fitness \
        --parallel \
        --num_proc 60 \
        --seed 29 \
        --informed \
        --resets 40-0.69
```

We summarized and visualized the results in the jupyter notebook `notebooks/demo_GB1.ipynb`. For an intuitive example illustrating the informed setting of MODIFY (the unique strength of our work), please also refer to this jupyter notebook.

## 3. Structure-based filtering

In MODIFY, we performed a quality check step during the library construction to ensure that the designed variants have good synthesizability. We used ESMFold pLDDT and FoldX $\Delta\Delta G$ as the metrics. For a detailed explanation, please refer to the jupyter notebook `notebooks/structure.ipynb` for our structure-based filtering. Specifically, we sampled from the designed library distribution and performed the filtering for library construction.

As the calculation of ESMFold and FoldX is time-consuming and computaton-intensive, we have uploaded a pre-calculated result `data/GB1/raw/GB1_structure.csv`. If you want to make the predictions on your own, we have provided the details of our implementation below.

(Optional) For FoldX $\Delta\Delta G$ calculation, you need to download FoldX from the official website (https://foldxsuite.crg.eu/), which may require you to apply for a license. For a given PDB structure (e.g., PDB:1PGA chain A for GB1), we first use the `RepairPDB` command of FoldX for an optimal performance from FoldX. Then, we use the `BuildModel` command to calculate $\Delta\Delta G$ with 5 repeated runs. We provide an example input mutant list for FoldX in `data/GB1/raw/individual_list.txt`.
```bash
# An example command for repairpdb, which requires the installation of FoldX
./foldx_20231231 --command=RepairPDB --pdb-dir=data/GB1 --pdb=1pga.pdb --output-dir=data/GB1/

# An example command for buildmodel, which requires the installation of FoldX and the translation of mutant name to FoldX required format (in individual_list.txt)
./foldx_20231231 --command=BuildModel --pdb=1pga_Repair.pdb --pdb-dir=data/GB1/ --mutant-file=data/GB1/raw/individual_list.txt --numberOfRuns=5 --output-dir=results/GB1/ --out-pdb=False

``` 

(Optional) For ESMFold pLDDT calculation, we first performed ESMFold predictions and then we extract pLDDT for each variant. We provided an example script for ESMFold predictions in `modify/esmfold.py`, which was modified from the official repo of ESMFold (https://github.com/facebookresearch/esm#esmfold), and also the function for calculating pLDDT, following the PymolFold repo (https://github.com/JinyuanSun/PymolFold). Need to use the `esmfold` conda environment.
```bash
python modify/esmfold.py -c 0 -s MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE -o results -n VDGV
```
And the function for ESMFold pLDDT extraction is as below.
```python
import biotite.structure.io as bsio

def cal_plddt(struct):
    plddt, cnt = 0, 0
    for s in struct:
        if s.atom_name=='CA':
            plddt += s.b_factor
            cnt +=1
    return plddt/cnt

struct = bsio.load_structure(f'results/VDGV.pdb', extra_fields=["b_factor"])
plddt = cal_plddt(struct)
```

## Contact

Please submit GitHub issues or contact Kerr Ding (kerrding[at]gatech[dot]edu) and Yunan Luo (yunan[at]gatech[dot]edu) for any questions related to the source code.
