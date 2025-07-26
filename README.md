<p align="center">
<img src="Figure 11.png" width="500" height="500">
</p>
<p align="center"><b>Figure: The model workflow</b></p>

# Introduction

Discovery of transcription factors (TFs) binding sites (TFBS) and their motifs in plants pose significant challenges due to high cross-species variability. The interaction between TFs and their binding sites is highly specific and context dependent. Most of the existing TFBS finding tools are not accurate enough to discover these binding sites in plants. They fail to capture the cross-species variability, interdependence between TF structure and its TFBS, and context specificity of binding. Since they are coupled to predefined TF specific model/matrix, they are highly vulnerable towards the volume and quality of data provided to build the motifs. All these software make a presumption that the user input would be specific to any particular TF which renders them of very limited uses. This all makes them hardly of any use for purposes like genomic annotations of newly sequenced species. Here, we report an explainable Deep Encoders-Decoders generative system, PTF-Vāc, founded on a universal model of deep co-learning on variability in binding sites and TF structure, PTFSpot, making it completely free from the bottlenecks mentioned above. It has successfully decoupled the process of TFBS discovery from the prior step of motif finding and requirement of TF specific motif models. Due to the universal model for TF:DNA interactions as its guide, it can discover the binding motifs in total independence from data volume, species and TF specific models. PTF-Vāc can accurately detect even the binding motifs for never seen before TF families and species, and can be used to define credible motifs from its TFBS report.

## 1. Environment setup

#### 1.1 Create and activate a new virtual environment

Users have their own choice of how to install required packages. But to efficiently manage the installation packages, Anaconda is recommended. After installing Annocoda, it would also be an good option to use virtual environment in annocoda. `conda activate` can be used to activate a virtual environment, and then install required packages. If users want to exit the virtual environment, simply type `conda deactivate`. 

#### 1.2 Install the package and other requirements

Run command to install pytorch

```
python3 -m pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U
```
Download and extract the source code for DeepBind and move to parent directory, type following commands:

```
unzip TF_protein.zip
```
#### 1.3 Software Requirements

***software list***
- python >=3.6
- pytorch
- numpy 
- pandas
- sklearn
- scipy 
- matplotlib

## 2. Data information

#### 2.1 Data processing
In this part, we will first introduce the **data information** used in this model, then introduce the training **data formats**, and finally introduce how to create a data set that meets to build the model requirements.

We have provided example data format compatible with DeepBind input data (DeepBind input data format: See `example/ABF2_pos.txt`. If you are trying to train DeepBind with your own data, please process your data into the same format as given in above example input data.

## 3. Model Training Based on Convolutional Neural Network (CNN)
#### 3.1 Training and testing 
**Input:** `ABF2_train.txt`, `ABF2_test.txt`. 
All data input files need to be placed in the same folder before training, such as in `example/` directory.

**Usage:**
Run the following command in the parent directory:
```
python3 deepbind.py ABF2
```
**Output:** 

**Final result** 
The trained model and best hyperparameter, `ABF2_Model.pth` and `ABF2_best_hyperpamarameters.pth`, are saved in the `output/` directory, respectively. 
The output file `ABF2_result.txt` located at `output/` directory contains the performance metrics of the test dataset.  

## Citation

If you use DeepBind in your research, please cite the following paper:</br>

"[PTF-Vāc: Ab-initio discovery of plant transcription factors binding sites using explainable and generative deep co-learning encoders-decoders](https://www.biorxiv.org/content/10.1101/2024.01.28.577608v2.full)",<br/>
bioRxiv.
