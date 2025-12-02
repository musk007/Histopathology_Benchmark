# How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark
## **Mohamed Bin Zayed University of Artificial Intelligence, Khalifa University and Shaukat Khanum Cancer Hospital**

![Benchmark Overview](overview.png)
We introduce Histo-VL, a fully open-source benchmark comprising images from 11 distinct acquisition tools, each with tailored captions incorporating class names and varied pathology descriptions. Histo-VL spans 26 organs and 31 cancer types, featuring tissue samples from 14 heterogeneous patient cohorts—totaling over 5 million patches from more than 41,000 WSIs at multiple magnification levels. We systematically evaluate histopathology visual-language models on Histo-VL to emulate expert tasks in clinical scenarios, providing a uniform framework to assess their real-world performance.
The following is a link to paper preprint [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2503.12990)

## Installation
1. Clone this repo and the [CONCH](https://github.com/mahmoodlab/CONCH) github repository.
2. Create an environment using the provided .yml file
```
conda env create -f benchmark.yml
conda activate benchmark
cd CONCH
pip install --upgrade pip
pip install -e .
pip install ../timm_ctp.tar --no-deps
```
3. Install requirements in the updated_requirements file.
4. Add the relative paths in :
    * The paths to the data, caching and results folder in the dotenv file : "plip/reproducibility/config_example.env"
    * The paths to the CONCH and MI-Zero models in : "plip/reproducibility/factory.py". The CONCH models can be downloaded from [here](https://huggingface.co/MahmoodLab/CONCH), and MI-Zero models from [here](https://github.com/mahmoodlab/MI-Zero)
    * Path to MI-Zero configuration path in : "plip/src/models/factory.py"
5. The captions for both available in the BenchCaption folder as well as the captions and splits for other experiments as well.
## Running Zero-shot
6. Run the following bash file in "plip/reproducibility" :
```
bash zero_shot.sh
```
7. To run experiments for the adversarial image or text corruptions or caption ensembling, you can specify that in the zero_shot.sh bash file or in plip/reproducibility/train.py
8. To create artifacts of a specific dataset, you can run the script at plip/reproducibility/generate_validation_datasets/artifactor.sh, which allows the creation of multiple types of artifacts, including blurring, rotation, addition of fat, squamous epithelial cells, and more.

## Datasets
   | Dataset Name | Dataset Link | Paper Link |
   |--------------|--------------|------------|
   | PanNuke    | [View Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) | [Read Paper1](https://arxiv.org/pdf/2003.10778),[Read Paper2](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2) |
   | CAMEL    | [View Dataset](https://github.com/ThoroughImages/CAMEL) | [Read Paper](https://arxiv.org/abs/1908.10555) |
   | Kather-16    | [View Dataset](https://zenodo.org/records/53169) | [Read Paper](https://www.nature.com/articles/srep27988) |
   | CRC-100K    | [View Dataset](https://zenodo.org/records/1214456) | [Read Paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002730) |
   | DigestPath (Kaggle)    | [View Dataset](https://www.kaggle.com/datasets/mittalswathi/digestpath-dataset) | - |
   | WSSS4LUAD    | [View Dataset](https://wsss4luad.grand-challenge.org/) | [Read Paper](https://arxiv.org/abs/2204.06455)  |
   | BACH    | [View Dataset](https://zenodo.org/records/3632035) | [Read Paper](https://arxiv.org/pdf/1808.04277)  |
   | BCNB    | [View Dataset](https://drive.google.com/drive/folders/1HcAgplKwbSZ7ZZl2m6PZdvVF70QJmVuR) | [Read Paper](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.759007/full))  |
   | BRACS    | [View Dataset](https://www.bracs.icar.cnr.it/download/) | [Read Paper](-)  |
   | BreakHist    | [View Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) | [Read Paper](https://ieeexplore.ieee.org/document/7312934)  |
   | Chaoyang    | [View Dataset](https://bupt-ai-cz.github.io/HSA-NRL/) | [Read Paper](https://ieeexplore.ieee.org/document/9600806)  |
   | TCGA-Uniform    | [View Dataset](https://zenodo.org/records/5889558#.YuJHdd_RaUk) | [Read Paper](https://www.nature.com/articles/ng.2764)  |
   | MHIST    | [View Dataset](https://bmirds.github.io/MHIST/) | [Read Paper](https://link.springer.com/chapter/10.1007/978-3-030-77211-6_2?utm_source=getftr&utm_medium=getftr&utm_campaign=getftr_pilot#Sec3)  |
   | CRC-TP    | [View Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/crc-tp) | [Read Paper](https://www.sciencedirect.com/science/article/pii/S136184152030061X#bib0045)  |
   | GlaS    | [View Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) | [Read Paper](https://arxiv.org/pdf/1603.00275v2)  |
   | LC25000    | [View Dataset](https://arxiv.org/pdf/1603.00275v2) | [Read Paper](https://arxiv.org/ftp/arxiv/papers/1912/1912.12142.pdf)  |
   | SICAPv2    | [View Dataset](https://data.mendeley.com/datasets/9xxm58dvs3/1) | [Read Paper](https://arxiv.org/pdf/2105.10490)  |
   | PCam    | [View Dataset](https://patchcamelyon.grand-challenge.org/Download/) | [Read Paper](https://arxiv.org/pdf/1806.03962)  |
   | CRC-ICM    | [View Dataset](https://data.mendeley.com/datasets/h3fhg9zr47/2) | [Read Paper](https://arxiv.org/abs/2308.10033)  |
   | DataBiox    | [View Dataset](https://databiox.com/) | [Read Paper](https://www.sciencedirect.com/science/article/pii/S2352914820300757)  |
   | Osteosarcoma    | [View Dataset](https://www.cancerimagingarchive.net/collection/osteosarcoma-tumor-assessment/) | [Read Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0210706)  |
   | PatchGastric    | [View Dataset](https://zenodo.org/records/6021442) | [Read Paper](https://arxiv.org/abs/2202.03432)  |
   | SkinCancer    | [View Dataset](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S) | [Read Paper](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.1022967/full)  |
   | RenalCell    | [View Dataset](https://data.niaid.nih.gov/resources?id=ZENODO_6528598) | [Read Paper](https://www.biorxiv.org/content/10.1101/2022.08.15.503955v1)  |
   | Breast-IDC    | [View Dataset](https://data.mendeley.com/datasets/hbdh66ws8d/1) | [Read Paper](-)  |
   | MPN (Ph-Negative Myeloproliferative Neoplasm.v2)    | [View Dataset](https://data.mendeley.com/datasets/hbdh66ws8d/1) | [Read Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10458278/)  |
   | TCGA-TIL    | [View Dataset](https://zenodo.org/records/6604094) | [Read Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5943714/)  |
   | MSI Classification (Snap Freeze)    | [View Dataset](https://zenodo.org/records/2532612#.Yt_Zdd_RZhE) | [Read Paper](http://doi.org/10.1016/j.immuno.2021.100008)  |
   | MSI Classification (FFPE)    | [View Dataset](https://zenodo.org/records/2530835) | [Read Paper](https://www.sciencedirect.com/science/article/pii/S1361841522001116?via%3Dihub)  |
   | Prostate Grading    | [View Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP) | [Read Paper](https://www.nature.com/articles/s41598-018-30535-1)  |
   | GasHisSDB    | [View Dataset](https://gitee.com/neuhwm/GasHisSDB#https://gitee.com/link?target=https%3A%2F%2Fdoi.org%2F10.6084%2Fm9.figshare.15066147.v1) | [Read Paper](-)  |
   | PCam    | [View Dataset]() | [Read Paper]()  |

## Current models included in the evaluation
| Model | Model | Paper Link |
   |--------------|--------------|------------|
   | PLIP    | [Model](https://github.com/PathologyFoundation/plip) | [Paper](https://www.nature.com/articles/s41591-023-02504-3)|
   | QuiltNet    | [Model](https://huggingface.co/wisdomik/QuiltNet-B-32) | [Paper](https://arxiv.org/abs/2306.11207) |
   | BiomedCLIP    | [Model](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | [Paper](https://arxiv.org/abs/2303.00915)|
   | Mi-Zero    | [Model](https://drive.google.com/drive/folders/1AR9agw2WLXes5wz26UTlT_mvJoUY38mQ) | [Paper](https://arxiv.org/abs/2306.07831) |
   | CONCH    | [Model](https://huggingface.co/MahmoodLab/CONCH) | [Paper](https://www.nature.com/articles/s41591-024-02856-4) |
   | KEEP    | [Model](https://huggingface.co/Astaxanthin/KEEP) | [Paper](https://arxiv.org/abs/2412.13126) |
   | PathCLIP    | [Model](https://huggingface.co/jamessyx/pathclip) | [Paper](https://arxiv.org/abs/2305.15072) |