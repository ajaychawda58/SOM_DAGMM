# SOM_DAGMM
Code for Self Organizing Map assissted Deep Autoencoding gaussian Mixture Model for intrusion detection

## Directory Structure

|	data
	|_____CreditCardFraud
		|_____creditcard.csv
	|_____NSL-KDD
		|_____*All files
	|_____arrhythmia.csv
|	som_dagmm
	|_____*Model Files
|	SOM.py
| 	train.py
|	eval.py
|	utils.py

## Data

Credit Card - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Arrhythmia - https://archive.ics.uci.edu/ml/datasets/arrhythmia
KDD - https://www.unb.ca/cic/datasets/nsl.html

## Training

train.py [-h] [--dataset DATASET] [--embedding EMBED] [--features FEATURES] [--embed EMBEDDING] [--batch_size BATCH_SIZE]

` python train.py --dataset kdd --embed label_encode --features numerical --epoch 100 --batch_size 1024`

## Evaluation

eval.py [-h] [--dataset DATASET] [--embedding EMBED] [--features FEATURES] [--threshold THRESHOLD]

` python eval.py --dataset kdd --embed label_encode --features numerical --threshold 20`

## References

1. Research Paper - https://arxiv.org/pdf/2008.12686.pdf
2. Code DAGMM - https://github.com/RomainSabathe/dagmm
3. Code SOM - https://github.com/JustGlowing/minisom