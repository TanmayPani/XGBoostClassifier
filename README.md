### XGBoostClassifier
A single C++ code that uses almost the full functionalities of the [xgboost](https://github.com/dmlc/xgboost/tree/master) library without depending on any bindings. This is a trimmed down version of the [xgboost-cli backend]( https://github.com/dmlc/xgboost/blob/master/src/cli_main.cc) which can be found here:

## Build instructions
All the following instructions are only tested for Ubuntu-22.04
### Pre-requisites
1) cmake (preferred)
2) pre built xgboost library
### Using CMake
If xgboost was built from source in a conda environment with `-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX` then                                                   ```
```
mkdir build 
cd build 
cmake ..
make
```
should work ().
A more robust way would be to pass the installation path of xgboost library as 
```
-DCMAKE_PREFIX_PATH=<\path\to\xgboost>
``` 
which, if xgboost was built from source using cmake, should be whatever path was passed as `CMAKE_INSTALL_PREFIX`. So the build process of this code should be:
```
mkdir build 
cd build 
cmake .. -DCMAKE_PREFIX_PATH=<\path\to\xgboost>
make
```
For an xgboost install using `pip` or `conda`, `<\path\to\xgboost>` should be path to the directory that contains the directory `lib` which should contain the static library `libxgboost.so` (`.dylib` for macOS I assume) . So the path to the static library should look like 
```
\path\to\xgboost\lib\libxgboost.so
```

## Usage
The code `XGBoostClassifier.cpp` performs training and prediction using the aptly names `Train` and `Predict` functions. The training and validation datasets are passed to `int main(...)` from the command line.

After compiling the code using `make`, 
```
cd ..
./build/XGBoostClassifier --help
```
To get info on all the arguments that can be passes to `int main(...)` function in  `XGBoostClassifier.cpp`. Information for most of the parameters for training using `xgboost::Learner` can be found here [here]([XGBoost Parameters — xgboost 2.0.3 documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)). 

The most essential arguments that are needed to run the code are,
1) `--training-data` : path to file containing training dataset.
2) `--validation-data` : paths to files containing validation datasets. Passed as successive arguments separated by spaces
3) `--format` : format of the datafiles `libsvm` or `csv`.
4) `--nrounds` : Maximum number of rounds of training.
5) `--metric-aggregation` : Specify how or if to aggregate the evaluations performed on the validation  datasets (can be `none, mean, best`  or  `all`).
6) `--eval-out-file` : File to which training and validation information is written.
7) `--metric-for-early-stopping` : Metric to watch for early stopping
8) `--early-stop-rounds` : Number of rounds to wait of the `--metric-for-early-stopping` to improve before training is stopped.
9) `--test-data-file` : Dataset on which to perform prediction. First validation dataset is taken for prediction if this argument is not passed.
10) `--pred-out-file` : File to which predictions are written.

refer to `runXGBClassifier.sh` scipt to see an example usage. To run the script on the example datasets provided, do,
```
bash runXGBClassifier.sh train.libsvm.txt test.libsvm.txt libsvm eval.txt preds.txt
```
In the terminal.
## Data format

The example datasets are in LIBSVM format where rows are written as,
```
label:weight  0:feature_0  1:feature_1  2:feature_2  ...  (nfeatures-1):feature_(nfeatures-1)
```

Refer [here]([Text Input Format of DMatrix — xgboost 2.0.3 documentation](https://xgboost.readthedocs.io/en/stable/tutorials/input_format.html)) for more information on different input formats.

## Outlook
Will add more information here after I refine the code a little more...
