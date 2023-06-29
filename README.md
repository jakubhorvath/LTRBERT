# 1. Installation

#### 1.1 Environment setup
```
conda create -n "LTR_BERT" python=3.8.16
conda activate LTR_BERT
pip3 install torch==1.10.1+cu113 torchvision>=0.11.1+cu113 torchaudio==0.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
```

### 1.2 Installing dependencies
```
pip install -r requirements.txt
```

# 2. Running 
The package is only a prototype at this stage and supports direct running from the ./Final_Models/predictors directory, however the trained models may be loaded and run anywhere.
To test a predictor {BERT_predictor.py, GBC_predictor.py, NN_predictor.py} run 
```
cd ./ltr_bert/Final_Models/predictors/
python BERT_predictor.py seq_file.fasta 
```

to test all models, the ./Final_Models/predictors/run_preds.ipynb notebooks is available. 
