# Real-Time Experimentation (RTX)

![Banner](https://raw.githubusercontent.com/Starofall/RTX/master/banner.PNG)


### Description
Real-Time Experimentation (RTX) tool allows for self-adaptation based on analysis of real time (streaming) data.
RTX is particularly useful in analyzing operational data in a Big Data environement.


### Minimal Setup
* Download the RTX code
* Run `python setup.py install` to download all dependencies 
* To run example experiments, first install [CrowdNav](https://github.com/Starofall/CrowdNav)
* To use Spark as a PreProcessor you also need to install Spark and set SPARK_HOME

### Getting Started Guide
A first guide is available at this [wiki page](https://github.com/Starofall/RTX/wiki/RTX-&-CrowdNav-Getting-Started-Guide)

### Abstractions

RTX has the following abstractions that can be implemented for any given service:
* PreProcessor - To handle Big Data volumes of data, this is used to reduce the volume
    * Example: Spark   
* DataProviders - A source of data to be used in an experiment
    * Example: KafkaDataProvider, HTTPRequestDataProvider
* ChangeProviders - Communicates experiment knobs/variables to the target system
    * Example: KafkaChangeProvider, HTTPRequestChangeProvider
* ExecutionStrategy - Define the process of an experiment
    * Example: Sequential, Gauss-Process-Self-Optimizing, Linear 
* ExperimentDefinition - A experiment is defined in a python file 
    * See `./experiment-specification/experiment.py`

### Supported execution strategies

* ExperimentsSeq - Runs a list of experiments one after another
    ```
    experiments_seq = [
        ...
        {
            "ignore_first_n_results": 100,
            "sample_size": 100,
            "knobs": {
                "exploration_percentage": 0.0
            }
        }
        ...
    ]
    ```


* SelfOptimizer - Runs multiple experiments and tries to find the best value for the knobs
    ```
    self_optimizer = {
        # Currently only Gauss Process
        "method": "gauss_process",
        # If new changes are not instantly visible, we want to ignore some results after state changes
        "ignore_first_n_results": 1000,
        # How many samples of data to receive for one run
        "sample_size": 1000,
        # The variables to modify
        "knobs": {
            # defines a [from-to] interval that will be used by the optimizer
            "max_speed_and_length_factor": [0.5, 1.5],
            "average_edge_duration_factor": [0.5, 1.5],
        }
    }
    ```
    
* StepExplorer - Goes through the ranges in steps (useful for graphs/heatmaps)
    ```
    step_explorer = {
        # If new changes are not instantly visible, we want to ignore some results after state changes
        "ignore_first_n_results": 10,
        # How many samples of data to receive for one run
        "sample_size": 10,
        # The variables to modify
        "knobs": {
            # defines a [from-to] interval and step
            "exploration_percentage": ([0.0, 0.2], 0.1),
            "freshness_cut_off_value": ([100, 400], 100)
        }
    }
    ```

### Extension for the investigation of Optimal Analysis propcess for the system

The experiment consists of the assumption for two output metric functions - Average overhead and Average feedback. The both metric functions are analysed to measure the system performance. Machine learning algorithms are used for this analysis. For Average Overhead the regression algorithms used are - Linear regression, polynomial regression and Decision tree. For Average feedback the classification algorithms used are - Logistic, Decision tree, Naive bayes, SVM and K-nearest neighbour. 

Finally, the optimal operating points are figured out by finding the optimal metric functions values.

* Analysis process embedded to the RTX execution

The analysis process is automated along with the RTX experiments. The process can be stopped by ignoring the relevant codes from the rtxlib\workflow.py file. The relevant code - 
   ```
   info(">start comparison of methods now")
   from compare_methods import regressor_compare_methods, classifier_compare_methods
   classifier_compare_methods()
   regressor_compare_methods()
   ```

* Input parameters' values 

The input parameters values can be provided for certain range in order to get a number of operating points among which the most optimal point can be found. The "definition.py" file holds the input parameters values within the dictionary "execution_strategy". The "knobs" array is considered to store the combination inputs variables values.The values can be updated according to the needs. It is expected to provide at least 10 number of input data points to achieve a satisfactory result.

```
   execution_strategy = {
    "ignore_first_n_results": 20,
    "sample_size": 20,
    "type": "sequential",
    "knobs": [
        {"route_random_sigma": 0.0,"exploration_percentage":0.0},
        {"route_random_sigma": 0.0,"exploration_percentage":0.1},
        {"route_random_sigma": 0.0,"exploration_percentage":0.2},
        {"route_random_sigma": 0.0,"exploration_percentage":0.3},
        {"route_random_sigma": 0.1,"exploration_percentage":0.0},
        {"route_random_sigma": 0.1,"exploration_percentage":0.1},
        {"route_random_sigma": 0.1,"exploration_percentage":0.2},
        {"route_random_sigma": 0.1,"exploration_percentage":0.3},
        {"route_random_sigma": 0.2,"exploration_percentage":0.0},
        {"route_random_sigma": 0.2,"exploration_percentage":0.1},
        {"route_random_sigma": 0.2,"exploration_percentage":0.2},
        {"route_random_sigma": 0.2,"exploration_percentage":0.3},
        {"route_random_sigma": 0.3,"exploration_percentage":0.0},
        {"route_random_sigma": 0.3,"exploration_percentage":0.1},
        {"route_random_sigma": 0.3,"exploration_percentage":0.2},
        {"route_random_sigma": 0.3,"exploration_percentage":0.3}
    ]
}

  ```

* results.csv

The output result from RTX experiment is stored into the files results.csv under the corresponding experiment folder (e.g. crowdnav-sequentials). Four columns are stored. First two columns represent two input variables - route_random_sigma and exploration_rate respectively. The last two columns represent two output metrics - Average feedback and Average overhead respectively.

```
0.0,0.0,4,2.7262119581425743

0.0,0.1,0,1.9971263285451974

0.0,0.2,0,2.1375472936468833

```

* Individual Machine learning methods

Within the folder machine learning models, all the algorithms are used for the analysis are listed. Each method can be executed individually with the comand - python 'filename'.py. Below are the notes to be considered for these methods. 

1) The files' list is - A) Classification: Logistic regression, KNN, Naive Bayes, Decision Tree, SVM.
                       B) Regression: - Linear regression, Polynomial and Decision Tree.
                       C) Others:- Ttest, outliers, Ttest-onetailed (Showed here only two models- logistic and SVM).

2) For Decision tree, by execution, the .dot files are generated in the same folder. Then those dot files should be translated using query - dot -Tpng F:\RTX-master\RTX-master/tree2.dot -o F:\RTX-master\RTX-master/tree2.png.

3) For ttest, the number of division of rows from results.csv is used very small number (Only 5) as we show only 10 number of data points. It can be increased for more number of result sets.

4) All the models can be embedded with the main RTX execution, only need to update the workflow file and the models' functionality.

