<!--
 * @Date: 2021-07-08 17:58:04
 * @LastEditors: yuhhong
 * @LastEditTime: 2021-07-13 16:21:55
-->
# Mass Spectrum Prediction from Molecular Fingerprint

This is an implementation of mass spectrum prediction from the [Extended Connectivity Fingerprint (ECFP)](https://pubs.acs.org/doi/10.1021/ci100050t) and the [Extended 3-Dimensional Fingerprint (E3FP)](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.7b00696) with a simple 5 layers' MLP model. The inputs are the SMILES strings of the molecules and the output is the MS2 level mass spectrum of the molecules. 

## Set up

Set up RDKit environment referring  https://www.rdkit.org/docs/Install.html.

```bash
conda create -c conda-forge -n rdkit-env rdkit
conda activate rdkit-env 
```



## Train & Test

The training and test datasets are split randomly with 9:1. 

```
usage: main.py [-h] [--device DEVICE] [--num_mlp_layers NUM_MLP_LAYERS]
               [--drop_ratio DROP_RATIO] [--batch_size BATCH_SIZE]
               [--in_dim IN_DIM] [--emb_dim EMB_DIM] [--out_dim OUT_DIM]
               [--train_subset] [--epochs EPOCHS] [--num_workers NUM_WORKERS]
               [--radius RADIUS] [--train_data_path TRAIN_DATA_PATH]
               [--test_data_path TEST_DATA_PATH] [--data_type DATA_TYPE]
               [--log_dir LOG_DIR] [--checkpoint_path CHECKPOINT_PATH]
               [--resume_path RESUME_PATH]
```

Command examples: 

```bash
# NIST17 
# (positive) 
python main.py --train_data_path ./data/NIST17/train_single_nist_msms_posi.sdf \
	--test_data_path ./data/NIST17/test_single_nist_msms_posi.sdf \
	--data_type sdf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/nist_posi.pt
# (negative) 
python main.py --train_data_path ./data/NIST17/train_single_nist_msms_nega.sdf \
	--test_data_path ./data/NIST17/test_single_nist_msms_nega.sdf \
	--data_type sdf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/nist_nega.pt

# GNPS 
# (positive) 
python main.py --out_dim 3000 --train_data_path ./data/GNPS/train_posi_main_GNPS.mgf \
	--test_data_path ./data/GNPS/test_posi_main_GNPS.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_posi.pt \
	--resume_path ./check_point/gnps_posi.pt 
python main.py --out_dim 3000 --train_data_path ./data/GNPS/train_posi_main_GNPS.mgf \
	--test_data_path ./data/GNPS/test_posi_main_GNPS.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_posi.pt \
	--resume_path ./check_point/gnps_posi.pt \
	--num_mlp_layers 12
# (negative) 
python main.py --out_dim 3000 --train_data_path ./data/GNPS/train_nega_main_GNPS.mgf \
	--test_data_path ./data/GNPS/test_nega_main_GNPS.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_nega.pt \
	--resume_path ./check_point/gnps_nega.pt
```



## Performance

| Dataset           | Fingerprint | Accuracy (Cosine Similarity on Validation) |
| ----------------- | ----------- | ------------------------------------------ |
| NIST17 (positive) | ECFP        | 0.3779                                     |
| NIST17 (negative) | ECFP        | 0.3066                                     |
| GNPS (positive)   | ECFP        | 0.3629                                     |
| GNPS (negative)   | ECFP        | 0.5146                                     |
| NIST17 (positive) | E3FP        |                                            |
| NIST17 (negative) | E3FP        |                                            |
| GNPS (positive)   | E3FP        |                                            |
| GNPS (negative)   | E3FP        |                                            |
