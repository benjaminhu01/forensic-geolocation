REM #!/bin/bash

SET PROJECT_DIR="C:/Users/Name/Documents/GitHub/forensic-geolocation"

SET FOLDS=10
SET PARTS=50

SET DOMAIN_PATH=%PROJECT_DIR%/data/external/global/domain.csv
SET CENTROIDS_PATH=%PROJECT_DIR%/data/external/global/country-centroids.csv

SET KNN_N_NEIGHBORS=25
SET RF_N_ESTIMATORS=200
SET NN_EPOCHS=5
SET NN_BATCH_SIZE=64
SET DNN_EPOCHS=5
SET DNN_BATCH_SIZE=64
SET AREA_CLF_EPOCHS=5
SET AREA_CLF_BATCH_SIZE=64

SET TAXA_THRESHOLD=0.0

python cross-validate.py data/processed/global/ITS.biom models/global/none-cross-val.csv ^
    --seeds 'none' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
    --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
    --centroids 'country' --centroids-fp %CENTROIDS_PATH% ^
    --area 'country' 'locality' 'administrative_area_level_1' ^
    'administrative_area_level_2' 'administrative_area_level_3' ^
    'administrative_area_level_4' 'administrative_area_level_5' ^
    --area-clf-epochs %AREA_CLF_EPOCHS% --area-clf-batch-size %AREA_CLF_BATCH_SIZE% ^
    --area-clf-verbose

rem rem python cross-validate.py data/processed/global/ITS.biom models/global/coarse-cross-val.csv ^
rem     --seeds 'coarse' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
rem     --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
rem     --area 'country' 'locality' 'administrative_area_level_1' ^
rem     'administrative_area_level_2' 'administrative_area_level_3' ^
rem     'administrative_area_level_4' 'administrative_area_level_5' ^
rem     --knn-n-neighbors %KNN_N_NEIGHBORS% --rf-n-estimators %RF_N_ESTIMATORS% ^
rem     --nn-epochs %NN_EPOCHS% --nn-batch-size %NN_BATCH_SIZE% --nn-verbose ^
rem     --dnn-epochs %DNN_EPOCHS% --dnn-batch-size %DNN_BATCH_SIZE% --dnn-verbose

rem python cross-validate.py data/processed/global/ITS.biom models/global/fine-cross-val.csv ^
rem     --seeds 'fine' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
rem     --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
rem     --area 'country' 'locality' 'administrative_area_level_1' ^
rem     'administrative_area_level_2' 'administrative_area_level_3' ^
rem     'administrative_area_level_4' 'administrative_area_level_5' ^
rem     --knn-n-neighbors %KNN_N_NEIGHBORS% --rf-n-estimators %RF_N_ESTIMATORS% ^
rem     --nn-epochs %NN_EPOCHS% --nn-batch-size %NN_BATCH_SIZE% --nn-verbose ^
rem     --dnn-epochs %DNN_EPOCHS% --dnn-batch-size %DNN_BATCH_SIZE% --dnn-verbose

rem python cross-validate.py data/processed/global/ITS.biom models/global/mixed-cross-val.csv ^
rem     --seeds='mixed' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
rem     --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
rem     --area 'country' 'locality' 'administrative_area_level_1' ^
rem     'administrative_area_level_2' 'administrative_area_level_3' ^
rem     'administrative_area_level_4' 'administrative_area_level_5' ^
rem     --knn-n-neighbors %KNN_N_NEIGHBORS% --rf-n-estimators %RF_N_ESTIMATORS% ^
rem     --nn-epochs %NN_EPOCHS% --nn-batch-size %NN_BATCH_SIZE% --nn-verbose ^
rem     --dnn-epochs %DNN_EPOCHS% --dnn-batch-size %DNN_BATCH_SIZE% --dnn-verbose

python cross-validate.py data/processed/global/100ITS.biom models/global/100-none-cross-val.csv \
    --seeds 'none' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
    --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
    --centroids 'country' --centroids-fp %CENTROIDS_PATH% ^
    --area 'country' 'locality' 'administrative_area_level_1' ^
    'administrative_area_level_2' 'administrative_area_level_3' ^
    'administrative_area_level_4' 'administrative_area_level_5' ^
    --area-clf-epochs %AREA_CLF_EPOCHS% --area-clf-batch-size %AREA_CLF_BATCH_SIZE% ^
    --area-clf-verbose

rem python cross-validate.py data/processed/global/100ITS.biom models/global/100-coarse-cross-val.csv ^
rem     --seeds='coarse' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
rem     --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
rem     --area country locality administrative_area_level_1 ^
rem     administrative_area_level_2 administrative_area_level_3 ^
rem     administrative_area_level_4 administrative_area_level_5 ^
rem     --knn-n-neighbors %KNN_N_NEIGHBORS% --rf-n-estimators %RF_N_ESTIMATORS% ^
rem     --nn-epochs %NN_EPOCHS% --nn-batch-size %NN_BATCH_SIZE% --nn-verbose ^
rem     --dnn-epochs %DNN_EPOCHS% --dnn-batch-size %DNN_BATCH_SIZE% --dnn-verbose

rem python cross-validate.py data/processed/global/100ITS.biom models/global/100-fine-cross-val.csv ^
rem     --seeds='fine' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
rem     --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
rem     --area 'country' 'locality' 'administrative_area_level_1' ^
rem     'administrative_area_level_2' 'administrative_area_level_3' ^
rem     'administrative_area_level_4' 'administrative_area_level_5' ^
rem     --knn-n-neighbors %KNN_N_NEIGHBORS% --rf-n-estimators %RF_N_ESTIMATORS% ^
rem     --nn-epochs %NN_EPOCHS% --nn-batch-size %NN_BATCH_SIZE% --nn-verbose ^
rem     --dnn-epochs %DNN_EPOCHS% --dnn-batch-size %DNN_BATCH_SIZE% --dnn-verbose

rem python cross-validate.py data/processed/global/100ITS.biom models/global/100-mixed-cross-val.csv ^
rem     --seeds='mixed' --folds %FOLDS% --partitions %PARTS% --domain-fp %DOMAIN_PATH% ^
rem     --region 0.5 0.75 0.9 --taxa-threshold %TAXA_THRESHOLD% ^
rem     --area 'country' 'locality' 'administrative_area_level_1' ^
rem     'administrative_area_level_2' 'administrative_area_level_3' ^
rem     'administrative_area_level_4' 'administrative_area_level_5' ^
rem     --knn-n-neighbors %KNN_N_NEIGHBORS% --rf-n-estimators %RF_N_ESTIMATORS% ^
rem     --nn-epochs %NN_EPOCHS% --nn-batch-size %NN_BATCH_SIZE% --nn-verbose ^
rem     --dnn-epochs %DNN_EPOCHS% --dnn-batch-size %DNN_BATCH_SIZE% --dnn-verbose
