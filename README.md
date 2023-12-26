# DARE

Implementation of: 
"DARE: development and validation of an attention based encoder for the analysis of diabetes routinely collected health-care data"

Dare is a BERT base model that can be easily fine-tuned for various clinical prediction tasks, enabling a personalized approach to the treatment of T2DM and advancing precision medicine for diabetes management. Details of the model can be found in ...


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


