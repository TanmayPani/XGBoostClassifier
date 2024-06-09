#!/bin/bash
TEST_DATA_LABEL="val"
DEVICE="cuda"
NROUNDS=1000
METRIC_AGGREGATION="none"
EARLY_STOPPING_ROUNDS=50
METRIC_FOR_EARLY_STOPPING="val-auc"

TRAINING_DATA="$1"
TEST_DATA="$2"
FORMAT="$3"
EVALHISTORY_FILE="$4"
PREDICTION_FILE="$5"

./build/XGBoostClassifier --training-data $TRAINING_DATA \
                          --validation-data $TEST_DATA \
                           --format $FORMAT \
                           --device $DEVICE \
                           --nrounds $NROUNDS \
                           --metric-aggregation $METRIC_AGGREGATION \
                           --eval-out-file $EVALHISTORY_FILE \
                           --early-stop-rounds $EARLY_STOPPING_ROUNDS \
                           --metric-for-early-stopping $METRIC_FOR_EARLY_STOPPING \
                           --pred-out-file $PREDICTION_FILE