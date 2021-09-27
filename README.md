<!--
 * @Date: 2021-07-08 17:58:04
 * @LastEditors: yuhhong
 * @LastEditTime: 2021-07-21 23:33:04
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
               [--radius RADIUS] [--fp_type FP_TYPE]
               [--train_data_path TRAIN_DATA_PATH]
               [--test_data_path TEST_DATA_PATH] [--data_type DATA_TYPE]
               [--log_dir LOG_DIR] [--checkpoint_path CHECKPOINT_PATH]
               [--resume_path RESUME_PATH]

GNN baselines on ogbgmol* data with Pytorch Geometrics

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       which gpu to use if any (default: 0)
  --num_mlp_layers NUM_MLP_LAYERS
                        number of mlp layers (default: 6)
  --drop_ratio DROP_RATIO
                        dropout ratio (default: 0.2)
  --batch_size BATCH_SIZE
                        input batch size for training (default: 256)
  --in_dim IN_DIM       input dimensionality (default: 1024)
  --emb_dim EMB_DIM     embedding dimensionality (default: 1600)
  --out_dim OUT_DIM     output dimensionality (default: 2000)
  --train_subset
  --epochs EPOCHS       number of epochs to train (default: 200)
  --num_workers NUM_WORKERS
                        number of workers (default: 0)
  --radius RADIUS       radius (default: 2)
  --fp_type FP_TYPE     fingerprint type [2d | 3d] (default: 2d)
  --train_data_path TRAIN_DATA_PATH
                        path to training data
  --test_data_path TEST_DATA_PATH
                        path to test data
  --data_type DATA_TYPE
                        type of dataset (sdf or mgf)
  --log_dir LOG_DIR     tensorboard log directory
  --checkpoint_path CHECKPOINT_PATH
                        path to save checkpoint
  --resume_path RESUME_PATH
                        path to resume checkpoint
```

Command examples: 

ECFP: 

```bash
# NIST17 
# (positive) 
python main.py --train_data_path ./data/NIST17/train_single_nist_msms_posi.sdf \
	--test_data_path ./data/NIST17/test_single_nist_msms_posi.sdf \
	--data_type sdf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/nist_posi.pt \
	--resume_path ./check_point/nist_posi.pt 
# (negative) 
python main.py --train_data_path ./data/NIST17/train_single_nist_msms_nega.sdf \
	--test_data_path ./data/NIST17/test_single_nist_msms_nega.sdf \
	--data_type sdf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/nist_nega.pt \
	--resume_path ./check_point/nist_nega.pt
# GNPS 
# (positive) 
python main.py --out_dim 3000 --train_data_path ./data/GNPS/train_ALL_GNPS_posi_high.mgf \
	--test_data_path ./data/GNPS/test_ALL_GNPS_posi_high.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_posi.pt \
	--resume_path ./check_point/gnps_posi.pt 
# (negative) 
python main.py --out_dim 3000 --train_data_path ./data/GNPS/train_ALL_GNPS_nega_high.mgf \
	--test_data_path ./data/GNPS/test_ALL_GNPS_nega_high.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_nega.pt \
	--resume_path ./check_point/gnps_nega.pt
```

E3FP: 

```bash
# NIST17 
# (positive) 
python main.py --fp_type 3d \
	--train_data_path ./data/NIST17/train_single_nist_msms_posi.sdf \
	--test_data_path ./data/NIST17/test_single_nist_msms_posi.sdf \
	--data_type sdf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/nist_posi_3d.pt \
	--resume_path ./check_point/nist_posi_3d.pt 
# (negative) 
python main.py --fp_type 3d \
	--train_data_path ./data/NIST17/train_single_nist_msms_nega.sdf \
	--test_data_path ./data/NIST17/test_single_nist_msms_nega.sdf \
	--data_type sdf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/nist_nega_3d.pt \
	--resume_path ./check_point/nist_nega_3d.pt
# GNPS 
# (positive) 
python main.py --fp_type 3d \
	--out_dim 3000 --train_data_path ./data/GNPS/train_posi_main_GNPS.mgf \
	--test_data_path ./data/GNPS/test_posi_main_GNPS.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_posi_3d.pt \
	--resume_path ./check_point/gnps_posi_3d.pt 
# (negative) 
python main.py --fp_type 3d \
	--out_dim 3000 --train_data_path ./data/GNPS/train_nega_main_GNPS.mgf \
	--test_data_path ./data/GNPS/test_nega_main_GNPS.mgf \
	--data_type mgf \
	--log_dir ./logs \
	--checkpoint_path ./check_point/gnps_nega_3d.pt \
	--resume_path ./check_point/gnps_nega_3d.pt
```

## Performance

| Dataset           | Fingerprint | Accuracy (Cosine Similarity on Validation) |
| ----------------- | ----------- | ------------------------------------------ |
| NIST17 (positive) | ECFP        | 0.4055                                     |
| NIST17 (negative) | ECFP        | 0.3364                                     |
| GNPS (positive)   | ECFP        | 0.3629                                     |
| GNPS (negative)   | ECFP        | 0.5146                                     |
| NIST17 (positive) | E3FP        |                                            |
| NIST17 (negative) | E3FP        |                                            |
| GNPS (positive)   | E3FP        |                                            |
| GNPS (negative)   | E3FP        |                                            |

