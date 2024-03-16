# eFold

This repo contains the pytorch code for our paper “*Diverse Database and Machine Learning Model to narrow the generalization gap in RNA structure prediction”* 

[[BioRXiv](https://www.biorxiv.org/content/10.1101/2024.01.24.577093v1.full)] [[Data](https://huggingface.co/rouskinlab)]

## Setup

### Download eFold

```bash
git clone https://github.com/rouskinlab/efold
```

### Create your environment

Using `virtualenv`

```bash
cd path/to/efold
python -m venv .venv
source .venv/bin/activate
pip install -m requirements.txt
```

Using conda

```bash
cd path/to/efold
conda create -n efold python=3.10
conda activate efold
pip install -m requirements.txt
```

### Install eFold

```bash
pip install path/to/efold
```

## File structure

```

bppm/ # bppm post processing step
efold/
    core/   # backend 
    models/ # where we define eFold and other models
scripts/
    efold_training.py # our training script
    [...]
best_efold/ # where we store our best model
    hyperparameters.json 
    weights.json
# python module boilerplate
LICENSE
requirements.txt
pyproject.toml
```

## Data

### List of the datasets we used

All the data is stored on the [Rouskin lab HuggingFace page](https://huggingface.co/rouskinlab). We use the following datasets:

**Pre-training:**

- `rnacentral_synthetic` : 226,729 sequences. RNAstructure prediction of a diverse RNA central subset.
- `ribo500-blast`: 46,060 sequences. [Ribonanza](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460121) sequences, predicted with RNAstructure, using Ribonanza’s chemical probing signal (CPS) as a constraint. Filtered out CPS < 500 reads and sequences with >80% match.
- `bpRNA` : 66,715 sequences. From [bpRNA](https://bprna.cgrb.oregonstate.edu/). Structure from covariance analysis.
- `RNAstralign` : 27,125 sequences. From [RNAstralign](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04540-7). Structure from covariance analysis.

**Fine-tuning**:

- `pri-miRNA` : 1,098 sequences. This data is new. Structure from RNAstructure using DMS-MaPseq chemical probing as a constraint.
- `human-mRNA` : 1,456 sequences. This data is new. Structure from RNAstructure using DMS-MaPseq chemical probing as a constraint.

**Testing**:

- `PDB` : 356 sequences. From [PDB](https://www.rcsb.org/stats/growth/growth-rna). Structure from NMR and crystallography.
- `viral-fragments` : 58 sequences. @Alberic Lajarte where is this data from?
- `lncRNA` : 15 sequences. [Where is this data from again?] @Alberic Lajarte .

### Get the data

You can download your datasets using [rouskinHF](https://github.com/rouskinlab/rouskinhf):

```bash
pip install rouskinhf
```

And in your code, write:

```python
>>> import rouskinhf
>>> data = rouskinhf.get_dataset('ribo500-blast')
```

## Inference mode

### Using the command line

From a sequence:

```bash
efold --sequence AACCTGGUG -o seq.txt
cat seq.txt
#TODO
```

or a fasta file:

```bash
efold --fasta example.fasta -o seq.txt
cat seq.txt
#TODO
```

### Using python

```python
>>> from efold import inference
>>> inference(seq = 'AACUGUGCUA')
#TODO
```

## Reproducing our results

Run the training script:

```bash
cd path/to/efold
python scripts/efold_training.py
```

## Citation

**Plain text:**

Albéric A. de Lajarte, Yves J. Martin des Taillades, Colin Kalicki, Federico Fuchs Wightman, Justin Aruda, Dragui Salazar, Matthew F. Allan, Casper L’Esperance-Kerckhoff, Alex Kashi, Fabrice Jossinet, Silvi Rouskin. “Diverse Database and Machine Learning Model to narrow the generalization gap in RNA structure prediction”. bioRxiv 2024.01.24.577093; doi: https://doi.org/10.1101/2024.01.24.577093. 2024

**BibTex:**

```
@article {Lajarte_Martin_2024,
	title = {Diverse Database and Machine Learning Model to narrow the generalization gap in RNA structure prediction},
	author = {Alb{\'e}ric A. de Lajarte and Yves J. Martin des Taillades and Colin Kalicki and Federico Fuchs Wightman and Justin Aruda and Dragui Salazar and Matthew F. Allan and Casper L{\textquoteright}Esperance-Kerckhoff and Alex Kashi and Fabrice Jossinet and Silvi Rouskin},
	year = {2024},
	doi = {10.1101/2024.01.24.577093},
	URL = {https://www.biorxiv.org/content/early/2024/01/25/2024.01.24.577093},
	journal = {bioRxiv}
}

```
