#include <dmlc/timer.h>

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/parameter.h>

#include <argparse/argparse.hpp>

#include <fstream>

using ParamDict_t = std::unordered_map<std::string, std::string>;
using EvalResult_t = std::unordered_map<std::string, std::vector<double>>;
using EvalHistory_t = std::unordered_map<std::string, std::vector<std::vector<double>>>;
using PredictTable_t = std::vector<std::vector<float>>;

enum class EvalMetricAggregationType{
    kNone = 0,
    kBest = 1,
    kMean = 2,
    kAll = 3
};

std::function<bool(float, float)> MetricImprovementCondition(std::string metricName, float& metricStartVal, bool verbose=true){
    static std::vector<std::string> decreasingMetrics = {"rmse", "logloss", "error", "merror", "mlogloss"};
    static std::vector<std::string> increasingMetrics = {"auc", "ndcg", "map"};
    static std::unordered_map<std::string, std::function<bool(float, float)>> metricImprovementConditions = {};

    metricName.erase(0, metricName.find('-')+1);
    auto metricModPos = metricName.find('@');
    if(metricModPos != metricName.npos)metricName.erase(metricModPos); //remove modifier if present (e.g. auc@0.5 -> auc

    if(metricImprovementConditions.find(metricName) != metricImprovementConditions.end()){
        if(verbose)std::cout<< "Improvement condition for metric: \""<<metricName<<"\" already determined"<<std::endl;
        return metricImprovementConditions[metricName];
    }

    bool isMetricTypeDecr = std::find(decreasingMetrics.begin(), decreasingMetrics.end(), metricName) != decreasingMetrics.end();
    bool isMetricTypeIncr = std::find(increasingMetrics.begin(), increasingMetrics.end(), metricName) != increasingMetrics.end();

    if(!isMetricTypeDecr && !isMetricTypeIncr){
        if(verbose)std::cout<< "Improvement condition not found for metric: \""<<metricName<<"\""<<std::endl;
        metricImprovementConditions[metricName] = [](float a, float b){return false;};
    }else if(isMetricTypeDecr){
        if(verbose)std::cout<< "Using improvement condition for decreasing metric: \""<<metricName<<"\""<<std::endl;
        metricStartVal = std::numeric_limits<float>::infinity();
        metricImprovementConditions[metricName] = [](float a, float b){return a < b;};
    }else if(isMetricTypeIncr){
        if(verbose)std::cout<< "Using improvement condition for increasing metric: \""<<metricName<<"\""<<std::endl;
        metricStartVal = -std::numeric_limits<float>::infinity();
        metricImprovementConditions[metricName] = [](float a, float b){return a > b;};
    }

    return metricImprovementConditions[metricName];
}

void AggregateVector(std::string metricName, std::vector<double>& vec, EvalMetricAggregationType aggType){
    if(aggType == EvalMetricAggregationType::kNone)return;
    float best;
    auto improvementCondition = MetricImprovementCondition(metricName, best, false);
    double sum = 0;
    double sumsqr = 0;
    double n = vec.size();
    for(auto& v : vec){
        if(improvementCondition(v, best))best = v;
        sum += v;
        sumsqr += v*v;
    }
    double mean = sum/n;
    double std = (n < 2) ? 0 : std::sqrt((sumsqr - n*mean*mean)/(n-1));

    if(aggType == EvalMetricAggregationType::kBest){
        vec = {best};
    }else if(aggType == EvalMetricAggregationType::kMean){
        vec = {mean, std};
    }else if(aggType == EvalMetricAggregationType::kAll){
        vec = {best, mean, std};
    }
}

EvalResult_t ParseEvalResult(std::string res, EvalMetricAggregationType aggType){
    res.erase(0, res.find('\t')+1);
    EvalResult_t evalRes;
    std::istringstream iss(res);
    std::string line;
    while(std::getline(iss, line, '\t')){
        std::istringstream lss(line);
        std::string token;
        std::vector<std::string> tokens;
        while(getline(lss, token, ':')){
            tokens.push_back(token);
        }
        std::string key = tokens[0];    
        double val = std::stof(tokens[1]);
        if(key.find("train") == key.npos){
            key.erase(0, key.find('-'));
            key = "val"+key;
        }
    
        if(evalRes.find(key) == evalRes.end()) evalRes[key] = {val};
        else evalRes[key].push_back(val);
    }

    for(auto& kv : evalRes){
        AggregateVector(kv.first, kv.second, aggType);
    }
    return evalRes;
}

bool EarlyStopCheck(unsigned int iteration, std::string metricName, EvalResult_t evalRes, unsigned int earlyStopRounds, unsigned int& bestIteration){
    static unsigned int nRoundsNoImprovement;
    static float bestMetricVal;
    static std::function<bool(float, float)> improvementCondition;

    if(iteration == 0){
        nRoundsNoImprovement = 0;
        improvementCondition = MetricImprovementCondition(metricName, bestMetricVal);
    }

    if(evalRes.find(metricName) == evalRes.end()){
        std::cout<< "Metric: "<<metricName<<" not found in evaluation results!"<<std::endl;
        return false;
    }

    float metricVal = evalRes[metricName][0];

    if(improvementCondition(metricVal, bestMetricVal)){
        std::cout<<metricName<<" improved from "<<bestMetricVal<<" to "<<metricVal<<" after "<<nRoundsNoImprovement+1<<" rounds"<<std::endl;
        nRoundsNoImprovement = 0;
        bestMetricVal = metricVal;
        bestIteration = iteration;
        std::cout<< "Best iteration so far: "<<iteration<<std::endl;
    }else{
        nRoundsNoImprovement++;
    }

    if(nRoundsNoImprovement > earlyStopRounds){
        std::cout<< "No improvement in "<<earlyStopRounds<<" rounds. "<<std::endl;
        std::cout<< "Early stopping triggered! ";
        std::cout<< "Best value of "<<metricName<<" "<<bestMetricVal<<" ";
        std::cout<< "Best iteration: "<<bestIteration<<std::endl;
        return true;
    }

    return false;
}

void SetParameters(xgboost::Learner* learner, ParamDict_t& params){
    static bool isPrinted = false;
    for(auto& kv : params){
        if(!isPrinted)std::cout<< "Setting parameter: "<<kv.first<<" = ";
        if(kv.first == "eval_metric"){
            std::istringstream iss(kv.second);
            std::string metric;
            while(std::getline(iss, metric, ' ')){
                if(!isPrinted)std::cout<< metric << " ";
                learner->SetParam("eval_metric", metric.c_str());
            }if(!isPrinted)std::cout<<std::endl;
        }else{
            if(!isPrinted)std::cout<< kv.second <<std::endl;
            learner->SetParam(kv.first.c_str(), kv.second.c_str());
        }
    }
    isPrinted = true;
}

EvalHistory_t Train(xgboost::Learner* learner, std::shared_ptr<xgboost::DMatrix> dmatTrain, std::vector<std::shared_ptr<xgboost::DMatrix>> evalDataSets, 
                    std::vector<std::string> evalNames, unsigned int nRounds, bool evalTrainDataSet, EvalMetricAggregationType aggType,
                    unsigned int earlyStopRounds, std::string metricForEarlystopping, unsigned int& bestIteration){
    unsigned int _bestIteration = 0;
    unsigned int nEvalDataSets = evalDataSets.size();
    if(evalTrainDataSet){
        evalDataSets.push_back(dmatTrain);
        evalNames.push_back("train");
    }

    bool doEarlyStopping = earlyStopRounds > 0 && metricForEarlystopping != "";

    EvalHistory_t evalHistory;
        
    for(unsigned int i = 0; i < nRounds; i++){
        learner->UpdateOneIter(i, dmatTrain);
        std::string res = learner->EvalOneIter(i, evalDataSets, evalNames);
        EvalResult_t evalRes = ParseEvalResult(res, aggType);
        std::cout<< "Evaluation result for round #" << i<<": ";   
        for(auto& kv : evalRes){
            evalHistory[kv.first].push_back(kv.second);
            std::cout<< kv.first << " = "<<kv.second[0]<<" ";
        }std::cout<<std::endl;
        if(doEarlyStopping){
            if(EarlyStopCheck(i, metricForEarlystopping, evalRes, earlyStopRounds, _bestIteration))break;
        }else{
            _bestIteration = i;
        }
    }
    bestIteration = _bestIteration;       
    return evalHistory; 
}

PredictTable_t Predict(xgboost::Learner* learner, std::shared_ptr<xgboost::DMatrix> dmatTest, unsigned int bestIteration){
    std::cout<< "Running Predictions ... "<<std::endl;
    std::cout<< "Model used from iteration: "<<bestIteration<<std::endl;

    unsigned long len, wlen, predlen;
    const float* trueTestLabels;
    const float* testSampleWeights;
    
    dmatTest->Info().GetInfo("label", &len, xgboost::DataType::kFloat32, reinterpret_cast<const void**>(&trueTestLabels));
    dmatTest->Info().GetInfo("weight", &wlen, xgboost::DataType::kFloat32, reinterpret_cast<const void**>(&testSampleWeights));
    xgboost::HostDeviceVector<float> predictions;
    learner->Predict(dmatTest, false, &predictions, 0, bestIteration+1);

    if(len != predictions.Size())std::cout<< "Length mismatch between true labels and predictions!"<<std::endl;

    PredictTable_t predTable;
    for(unsigned long i = 0; i < len; i++){
        predTable.push_back({trueTestLabels[i], predictions.ConstHostVector()[i], testSampleWeights[i]});
    }

    return predTable;
}

void WriteEvalHistory(std::string outFile, EvalHistory_t& evalHistory, EvalMetricAggregationType aggType){
    std::ofstream out(outFile.c_str());
    out<<"Aggregation type: "<<int(aggType)<<std::endl;
    out<<"Columns: ";
    for(auto& kv : evalHistory){
        for(auto& v : kv.second[0]){
            out<<kv.first<<" ";
        }
    }out<<std::endl;
    int nRounds = evalHistory.begin()->second.size();
    auto it = evalHistory.begin();
    for(int i = 0; i < nRounds; i++){
        for(auto& kv : evalHistory){
            for(auto& v : kv.second[i]){
                out<<v<<" ";
            }
        }out<<std::endl;
    }
    out.close();
    std::cout<< "Evaluation history written to: "<<outFile<<std::endl;
}

void WritePredictions(std::string outFile, PredictTable_t& predTable){
    std::ofstream out(outFile.c_str());
    for(auto& row : predTable){
        for(auto& v : row){
            out<<v<<" ";
        }out<<std::endl;
    }out.close();
    std::cout<< "Predictions written to: "<<outFile<<std::endl;
}

void Run(std::string trainDataURI, std::vector<std::string> valDataURIs, std::vector<std::string>valDataNames, ParamDict_t& params, unsigned int nRounds, 
        EvalMetricAggregationType aggType, std::string testDataURI, std::string evalHistOutFile, std::string predOutFile,
        unsigned int earlyStopRounds=0, std::string metricForEarlystopping=""){
    std::shared_ptr<xgboost::DMatrix> dmatTrain(xgboost::DMatrix::Load(trainDataURI, false, xgboost::DataSplitMode::kRow));
    std::vector<std::shared_ptr<xgboost::DMatrix>> cachedDmats = {dmatTrain};
    std::vector<std::shared_ptr<xgboost::DMatrix>> dmatsVal;
    for(auto& valDataURI : valDataURIs){
        cachedDmats.emplace_back(xgboost::DMatrix::Load(valDataURI, false, xgboost::DataSplitMode::kRow));
        dmatsVal.push_back(cachedDmats.back());
    }

    std::unique_ptr<xgboost::Learner> learner(xgboost::Learner::Create(cachedDmats));
    SetParameters(learner.get(), params);

    unsigned int bestIteration = 0;
    auto history = Train(learner.get(), dmatTrain, dmatsVal, valDataNames, nRounds, true, aggType, earlyStopRounds, metricForEarlystopping, bestIteration);

    if(evalHistOutFile != "")WriteEvalHistory(evalHistOutFile, history, aggType);

    std::shared_ptr<xgboost::DMatrix> dmatTest(xgboost::DMatrix::Load(testDataURI, false, xgboost::DataSplitMode::kRow));

    auto predictions = Predict(learner.get(), dmatTest, bestIteration);
    if(predOutFile != "")WritePredictions(predOutFile, predictions);

}

void ParseMainArguments(int argc, char *argv[], argparse::ArgumentParser &parser){
    parser.add_argument("--training-data").required()
        .help("Path to training data file");
    parser.add_argument("--validation-data").required().nargs(argparse::nargs_pattern::at_least_one)
        .help("Paths to validation data files");
    parser.add_argument("--format").default_value(std::string("libsvm"))
        .help("Data format (libsvm or csv)");

    parser.add_argument("--device").default_value(std::string("cpu"))
        .help("Device to use for training (cpu or cuda)");
    parser.add_argument("--booster").default_value(std::string("gbtree"))
        .help("Type of booster to use");
    parser.add_argument("--tree-method").default_value(std::string("hist"))
        .help("Tree construction method");
    parser.add_argument("--subsample").default_value(std::string("0.8"))
        .help("Subsample ratio of the training instances");
    parser.add_argument("--objective").default_value(std::string("binary:logistic"))
        .help("Objective function");
    parser.add_argument("--eval-metric").default_value(std::string("auc error"))
        .help("Evaluation metrics");
    parser.add_argument("--num-parallel-tree").default_value(std::string("10"))
        .help("Number of parallel trees constructed during each iteration");
    parser.add_argument("--colsample-bynode").default_value(std::string("0.8"))
        .help("Subsample ratio of columns when constructing each tree");
    parser.add_argument("--reg-lambda").default_value(std::string("1e-5"))
        .help("L2 regularization term on weights");
    parser.add_argument("--verbosity").default_value(std::string("0"))
        .help("Verbosity of messages");
    parser.add_argument("--eta").default_value(std::string("1"))
        .help("Step size shrinkage used in update to prevent overfitting");
    parser.add_argument("--max-depth").default_value(std::string("6"))
        .help("Maximum depth of a tree");
    parser.add_argument("--max-bin").default_value(std::string("1024"))
        .help("Maximum number of bins that the histogram construction algorithm will use");
    parser.add_argument("--min-child-weight").default_value(std::string("1"))
        .help("Minimum sum of instance weight needed in a child");
    parser.add_argument("--base-score").default_value(std::string("0.5"))
        .help("Initial prediction score of all instances");
    parser.add_argument("--validate-parameters").default_value(std::string("true"))
        .help("Whether to validate parameters");

    parser.add_argument("--nrounds").default_value(10).scan<'i', int>()
        .help("Number of boosting rounds");
    parser.add_argument("--metric-aggregation").default_value(std::string("mean"))
        .help("Type of aggregation for evaluation metrics (none, best, mean, all)");
    parser.add_argument("--eval-out-file").default_value(std::string(""))
        .help("Path to test data file");

    parser.add_argument("--early-stop-rounds").default_value(0).scan<'i', int>()
        .help("Number of rounds with no improvement after which training will stop");
    parser.add_argument("--metric-for-early-stopping").default_value(std::string(""))
        .help("Metric to use for early stopping");

    parser.add_argument("--test-data-file").default_value(std::string(""))
        .help("Path to test data file");
    parser.add_argument("--pred-out-file").default_value(std::string(""))
        .help("Output file for predictions");

    parser.parse_args(argc, argv);
}
    

int main(int argc, char *argv[]){
    argparse::ArgumentParser parser("XGBoostClassifier");
    ParseMainArguments(argc, argv, parser);

    ParamDict_t params;
    std::string device = parser.get<std::string>("--device");
    params["device"] = device;
    params["sampling_method"]  = (device == "cuda") ? "gradient_based" : "uniform";
    params["booster"]          = parser.get<std::string>("--booster");  
    params["tree_method"]      = parser.get<std::string>("--tree-method");
    params["subsample"]        = parser.get<std::string>("--subsample");
    params["objective"]        = parser.get<std::string>("--objective");
    params["eval_metric"]      = parser.get<std::string>("--eval-metric");
    params["num_parallel_tree"] = parser.get<std::string>("--num-parallel-tree");
    params["colsample_bynode"] = parser.get<std::string>("--colsample-bynode");
    params["reg_lambda"]       = parser.get<std::string>("--reg-lambda");
    params["verbosity"]        = parser.get<std::string>("--verbosity");
    params["eta"]              = parser.get<std::string>("--eta");
    params["max_depth"]        = parser.get<std::string>("--max-depth");
    params["max_bin"]          = parser.get<std::string>("--max-bin");
    params["min_child_weight"] = parser.get<std::string>("--min-child-weight");              
    params["base_score"]       = parser.get<std::string>("--base-score");
    params["validate_parameters"] = parser.get<std::string>("--validate-parameters");

    std::string trainDataPath = parser.get<std::string>("--training-data");
    std::cout<< "Training on: "<<trainDataPath<<std::endl;

    std::string format = parser.get<std::string>("--format");
    std::string trainDataURI = trainDataPath+"?format="+format;

    std::vector<std::string> valDataPaths = parser.get<std::vector<std::string>>("--validation-data");
    std::vector<std::string> valDataLabels;
    std::vector<std::string> valDataURIs;
    std::cout<< "Validating on: "<<std::endl;
    int i = 0;
    for(auto& filePath : valDataPaths){
        valDataLabels.push_back("val"+std::to_string(i));
        std::cout<< filePath <<" "<<valDataLabels[i++]<<std::endl;
        valDataURIs.push_back(filePath+"?format="+format);
    }

    unsigned int nrounds = parser.get<int>("--nrounds"); 

    std::string aggTypeStr = parser.get<std::string>("--metric-aggregation");
    EvalMetricAggregationType aggType;
    if(aggTypeStr == "none")aggType = EvalMetricAggregationType::kNone;
    else if(aggTypeStr == "best")aggType = EvalMetricAggregationType::kBest;
    else if(aggTypeStr == "mean")aggType = EvalMetricAggregationType::kMean;
    else if(aggTypeStr == "all")aggType = EvalMetricAggregationType::kAll;

    std::string evalHistOutFile = parser.get<std::string>("--eval-out-file");

    unsigned int earlyStopRounds = parser.get<int>("--early-stop-rounds");
    std::string metricForEarlystopping = parser.get<std::string>("--metric-for-early-stopping");

    std::string testDataPath = parser.get<std::string>("--test-data-file");
    std::string predOutFile = parser.get<std::string>("--pred-out-file");

    std::string testDataURI = "";
    if(predOutFile != ""){
        if(testDataPath == "")testDataPath = valDataPaths[0];
        testDataURI += testDataPath+"?format="+format;
    }

    Run(trainDataURI, valDataURIs, valDataLabels, params, nrounds, aggType, testDataURI, evalHistOutFile, predOutFile, earlyStopRounds, metricForEarlystopping);
    std::cout<< "---------------------------------------------"<<std::endl; 

    return 0;
}


