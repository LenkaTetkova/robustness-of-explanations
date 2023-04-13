# Robustness of Visual Explanations to Common Data Augmentation Methods
==============================
Code for reproducing the paper Lenka Tětková and Lars Kai Hansen: **Robustness of Visual Explanations to Common Data Augmentation Methods** (The 2nd Explainable AI for Computer Vision (XAI4CV) Workshop at CVPR 2023)

------------
## How to use
1. Create Conda environment from the environment file
`conda env create -f environment.yml`
or use the requirements file requirements.txt.

2. Download the validation set of imagenet-mini (https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) and place it to the data folder (the code expects the path `f"./data/imagenet-mini/val/{class ID}/{figure name}"`.

3. Run
```
chmod +x jobscript.sh
source jobscript.sh
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
