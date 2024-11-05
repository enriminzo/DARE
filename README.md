# DARE

Implementation of: 
"[A Deep Attention-Based Encoder for the Prediction of Type 2 Diabetes Longitudinal Outcomes from Routinely Collected Health Care Data](https://doi.org/10.1101/2024.11.02.24316561)"

Dare is a transformer base encoder that can be easily fine-tuned for various clinical prediction tasks, enabling a personalized approach to the treatment of T2DM and advancing precision medicine for diabetes management. 


## Get Started

To get started you first need to install [PyTorch](https://pytorch.org/get-started/locally/).
Next, clone and install with:

```sh
git clone https://github.com/enriminzo/DARE.git
cd DARE
pip install .
```

## Data preprocessing
Before using the code you can set data paths in `./Dare/Condfigs/data_paths.yaml`
You can find the code for data preprocessing in `./Preprocessing`. These scripts work with data extracted from the SIDIAP database, but it can be easily adapted to different data structures. 

## Usage of DARE
The code in `Pretrain_parameters_optimization.py` can be used to search optimal parameters and get the pretrained models. To Fine-tune the models for a specific clinical task use the script `FineTune_model.py`

## Citation
If you find this code or our article usefull for your work you can cite it:

@article {Manzini2024,<br/> 
author = {Manzini, Enrico and Vlacho, Bogdan and Franch-Nadal, Josep and Escudero, Joan and Génova, Ana and Andrés, Eric and Reixach, Elisenda and Pizarro, Israel and Mauricio, Dídac and Perera-LLuna, Alexandre},<br/> 
title = {A Deep Attention-Based Encoder for the Prediction of Type 2 Diabetes Longitudinal Outcomes from Routinely Collected Health Care Data},<br/> 
year = {2024},<br/> 
doi = {10.1101/2024.11.02.24316561},<br/> 
publisher = {Cold Spring Harbor Laboratory Press},<br/> 
URL = {https://www.medrxiv.org/content/early/2024/11/04/2024.11.02.24316561},<br/> 
eprint = {https://www.medrxiv.org/content/early/2024/11/04/2024.11.02.24316561.full.pdf},<br/> 
journal = {medRxiv}<br/> 
}

