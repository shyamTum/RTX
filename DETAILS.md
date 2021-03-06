# Self-Adaptation Based on Big Data Analytics : A Model Problem and Tool  
## Overview
In this document we give a brief introduction to two two artifacts we have bundled together.

![Banner](https://raw.githubusercontent.com/Starofall/CrowdNav/master/banner.PNG)


* **CrowdNav** - a simulation based on SUMO and TraCI that implements a custom router that can be configured using kafka messages or local JSON config on the fly while the simulation is running. Also runtime data is send to a kafka queue to allow stream processing and logger locally to CSV.

![Banner](https://raw.githubusercontent.com/Starofall/RTX/master/banner.PNG)

* **RTX** - a tool that allows for self-adaptation based on analysis of real time (streaming) data. RTX is particularly useful in analyzing operational data in a Big Data environment.

## Getting started guide

To simplify working with the artifacts, we offer multiple ways to run them and
also worked hard to allow a cross-platform usage on Windows, Linux and MacOS.
 
You have the following options when running the artifact:

* ##### Running through docker
The easiest way to have a look into the tool and what it is doing is to use our provided docker containers.

To use this approach follow the following instructions:
1) Get **Docker** running on your OS - [Click for Instructions](https://docs.docker.com/engine/installation/)

2) The next step is to run **Kafka** as we use this instance for CrowdNav and RTX.
To do this execute the following command. It will start [pre-configured](https://hub.docker.com/r/spotify/kafka/) kafka server with the hostname kafka and open the relevant ports.
    ```
    docker run --name kafka --hostname kafka -p 2181:2181 -p 9092:9092 --env ADVERTISED_HOST=kafka --env ADVERTISED_PORT=9092 spotify/kafka
    ```
3) Once kafka is running we start **CrowdNav** to run the simulation. Here we offer two options:
    * console mode - where CrowdNav works using console output as long as the container is alive.
    ```
    docker run -it --link kafka:kafka starofall/crowdnav
    ```
    
    * gui mode - to see what CrowdNav is doing, this mode allows the user to see the SUMO-GUI graphical user interface using VNC. 
    ```
    docker run -it -p 6080:6080 --link kafka:kafka starofall/crowdnavgui-docker
    ```
    Once the container is running, open the browser to either `localhost:6080` or `(IP of your Boot2Docker):6080`.
    Here you should see the desktop transmitted over vnc. Once you click the shortcut on the desktop, CrowdNav boots up and the simulation can be started using the play button.

* ##### Running through VM
If you want to take a deeper look into the tool and also want to play around with the configuration files, we provide a VM image that is already pre-configured to have all dependencies and software installed.

1) Download and import the provided VHD file into your VM provider. Give the VM at least 1GB of RAM to run. 
2) Once the VM is running you have to start kafka using the icon on the desktop. Always click on "open in terminal" and if asked the password is "seam".
3) RTX and CrowdNav are already provided on the desktop with additional tooling scripts. 
It is recommended to read the readme files provided in these folders. Also run the update scripts to assure you are using the most recent version.
4) You can run CrowdNav without doing experiments using RTX, but not vice versa.

* ##### Setting up and running locally
If you want to use the tool more extensively it is recommended to install it locally. 

To do so follow these instructions:
1) For Kafka we also propose to use the docker container as described in the text above. 
   When you are using Boot2Docker you might need to add an entry to your hosts file to link `(IP of your Boot2Docker)` to the hostname `kafka`.
   If you want to install Kafka manually, please read [these instructions](https://kafka.apache.org/quickstart).
2) Next step is to install SUMO. The instructions for this are depending on your operating system:
    * [Official Instructions](http://sumo.dlr.de/wiki/Installing)
    * Please make sure you at least have version 0.27 installed and the env variable `SUMO_HOME` is set to the sumo folder.
    * If your distro does offer an older version than 0.27, you have to build SUMO from source using this commands:
        ```
        sudo apt-get install -y build-essential git libxerces-c-dev
        sudo mkdir -p /opt && sudo cd /opt
        sudo git clone https://github.com/radiganm/sumo.git
        sudo cd /opt/sumo && sudo ./configure
        sudo cd /opt/sumo && sudo make
        sudo cd /opt/sumo && sudo make install
        export SUMO_HOME=/opt/sumo
        ```
3) Extract CrowdNav from the archive or download the recent source code from the [CrowdNav Repository](https://github.com/Starofall/CrowdNav).
4) Make sure you have `python 2.7` installed.
5) Run the following commands within the CrowdNav folder: 
    ```
    sudo apt-get install libfreetype6-dev libpng-dev # needed for older distros
    sudo python setup.py install
    ```
6) Now you should be able to run `python run.py` and see the SUMO interface where you have to start the simulation.
7) Extract RTX from the archive or download the recent source code from the [RTX Repository](https://github.com/Starofall/RTX).
8) On Windows you have to manually install `numpy` and `scipy`. We provide compiled versions of these libraries and an install script for this at `tools/winlibs/install.bat`.
9) Also on older distros installing `numpy` and `scipy` can be challenging. Please look [here](https://www.scipy.org/install.html) and [here](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html) for more information.  
10) Run the following commands within the RTX folder: 
    ```
    sudo python setup.py install
    ```
11) If everything worked, you can now run experiments - e.g. by using:
    ```
    python rtx.py start crowdnav-sequential
    ```       

###### Special Notes
* CrowdNav will not send messages for the first 1000 ticks to let the system stabilize, so please be patient.
* Running a RTX experiment that is using Spark can also take some time to bootup.  

## Step-by-step instructions

Once you have the system installed you can do the following things:

##### Within CrowdNav
* Run the GUI application and inspect that there are cars that are red (controlled by us) and green (send to exploration by us).
* See where traffic jams happen and how good the red cars are avoiding them over the time.
* Modify the source code of the CustomRouter.py to your own routing algorithm.
* If you set `kafkaUpdates` to `False` in the `Config.py` file you can adjust the router with the `knobs.json` file.
    * Do not forget to revert this if you want to use RTX for running experiments.
* Use `forever.py` to run the simulation in the background forever.
* Use `parallel.py #n` to spawn n processes and let them run for 10.000 ticks for benchmarks and data generation.
##### Within RTX
* Try out all the provided examples (see list of folders starting with crowdnav) by issuing

        python rtx.py start [folder name]
        
* Look at the auto generated diagrams for the experiments 
* Regenerate the diagrams from already run experiments by issuing

        python rtx.py report [folder name]

* Change variables in the different `definition.py` to run other experiments.
* Implement a custom target system and optimize it.
 
## Further details

### RTX - Abstractions

RTX has the following abstractions that can be implemented for any given service:
* PreProcessor - Is used to reduce the volume of Big Data streams.
    * Example: Spark   
* DataProviders - Defines a source of data to be used in an experiment
    * Example: KafkaDataProvider
* ChangeProviders - Communicates experiment knobs/variables to the target system
    * Example: KafkaChangeProvider
* ExecutionStrategy - Defines the process of an experiment
    * Example: Sequential, Gauss-Process-Self-Optimizing, Step 

### RTX - `definition.py`

Experiments in RTX are defined in a seperate folder using a file named `definition.py`. Here the user can define how an experiment works. It consists of the following segments.

```
name = ""
```
Each experiment gets a name that is used to identify itself.
```
system = {
    # Defines how to run experiments
    # "sequential"     -> Runs a list of experiments in a sequential way
    #    requires a "experiments_seq" array in the definition.py
    # "self_optimizer" -> Runs a self adaptation algorithm to optimize values
    #    requires a "self_optimizer" object in the definition.py
    # "step"           -> Goes through the range in steps (even on two dimensions)
    #    requires a "step_explorer" object in the definition.py
    "execution_strategy": "",

    # We can install a preprocessor like Spark to reduce data volume
    # "spark" -> Submits a preprocessor to spark to reduce the message volume
    #    requires a "spark" element in the configuration section
    # "none"  -> We directly connect to the data source and do not use a preprocessor
    "pre_processor": "",

    # What provider we use to get data from the running experiments
    # "kafka_consumer" -> Gathers data through listening to a kafka topic
    #    requires a "kafkaConsumer" element in the configuration section
    # "mqtt_listener" -> Gathers data from a MQTT queue
    #    Not yet implemented
    # "http_data_requests" -> Gathers data from doing active http requests to the system
    #    Not yet implemented
    "data_provider": "",

    # What provider we use to change the running experiment
    # "kafka_producer" -> Doing changes by pushing to kafka
    #    requires a "kafkaProducer" element in the configuration section
    # "mqtt_publisher" -> Doing changes by pushing to mqtt
    #    Not yet implemented
    # "http_change_requests" -> Doing changes by calling a http interface
    #    Not yet implemented
    "change_provider": "",

    # Initializes a new state for an experiment
    #   definition: (empty_dict) => init_state
    "state_initializer": lambda empty_dict: {},

    # All incoming streaming data are reduced
    #   definition: (old_state,new_data) => new_state
    "data_reducer": lambda old_state, new_data: {},

    # The evaluation function that evaluates this experiment
    # Auto optimizing is trying to minimize this value
    #   definition: (result_state) => float
    "evaluator": lambda result_state: 0.0,

    # As variables change in the run, this function is used to generate the input
    # of the change provider to apply the new variable.
    #   definition: (variables) => input_for_change_provider
    "change_event_creator": lambda result_state: {}
}
```
Here the user can define settings of how he wants to run this experiment.
Expecially important are the four functions that the user has to provide to the system to adapt it to its specific target environment - you can see our examples for this in the provided crowdnav experiment folders.
```

# Defines the settings for the modules used in the workflow
configuration = {
    # If we use the Spark preprocessor, we have to define this sparkConfig
    "spark": {
        # currently we only support "local_jar"
        "submit_mode": "",
        # name of the spark jobs jar (located in the experiment's folder) - e.g. "assembly-1.0.jar"
        "job_file": "",
        # the class of the script to start - e.g. "crowdnav.Main"
        "job_class": ""
    },
    # If we use KafkaProducer as a ChangeProvider, we have to define this kafkaProducerConfig
    "kafka_producer": {
        # Where we can connect to kafka - e.g. kafka:9092
        "kafka_uri": "",
        # The topic to listen to
        "topic": "",
        # The serializer we want to use for kafka messages
        #   Currently only "JSON" is supported
        "serializer": "",
    },
    # If we use KafkaConsumer as a DataProvider, we have to define this kafkaConsumerConfig
    "kafka_consumer": {
        # Where we can connect to kafka
        "kafka_uri": "",
        # The topic to listen to
        "topic": "",
        # The serializer we want to use for kafka messages
        #   Currently only "JSON" is supported
        "serializer": "",
    },
}
```
Here the user has to tell RTX the configuration values for the selected providers. If he wants to use spark he also has to add the job_file into the experiments folder and set the `job_file` and `job_class` correctly.
```
# If we use ExecutionStrategy "self_optimizer" ->
self_optimizer = {
    # Currently only "gauss_process" is supported
    "method": "",
    # If new changes are not instantly visible, we want to ignore some results after state changes
    "ignore_first_n_results": 1000,
    # How many samples of data to receive for one run
    "sample_size": 1000,
    # The variables to modify
    "knobs": {
        # defines a [from-to] interval that will be used by the optimizer
        "variable_name": [0.0, 1.0]
    }
}
```
If the user wants to use the `self_optimizer` strategy, this part is needed to tell RTX which knob should be optimized and in which range (here we optimize `variable_name` in the range from 0 to 1).
```
# If we use ExecutionStrategy "sequential" ->
experiments_seq = [
    {
        # Variable that is changed in the process
        "knobs": {
            "variable_name": 0.0
        },
        # If new changes are not instantly visible, we want to ignore some results after state changes
        "ignore_first_n_results": 1000,
        # How many samples of data to receive for one run
        "sample_size": 1000,
    },
    {
        # Variable that is changed in the process
        "knobs": {
            "variable_name": 0.1
        },
        # If new changes are not instantly visible, we want to ignore some results after state changes
        "ignore_first_n_results": 1000,
        # How many samples of data to receive for one run
        "sample_size": 1000,
    }
]
```
If the user wants to use the `sequential` strategy, this part is needed to tell RTX which experiments should get executed.
```
# If we use ExecutionStrategy "step" ->
step_explorer = {
    # If new changes are not instantly visible, we want to ignore some results after state changes
    "ignore_first_n_results": 10,
    # How many samples of data to receive for one run
    "sample_size": 10,
    # The variables to modify
    "knobs": {
        # defines a [from-to] interval and step
        "variable_name": ([0.0, 0.4], 0.1),
    }
}
```
If the user wants to use the `step` strategy, this part is needed to tell RTX which knob values should be tested and in which range (here we test `variable_name` from 0.0 to 0.4 in steps of 0.1)


### Extension for the investigation of Optimal Analysis propcess for the system

The experiment consists of the assumption for two output metric functions - Average overhead and Average feedback. The both metric functions are analysed to measure the system performance. Machine learning algorithms are used for this analysis. For Average Overhead the regression algorithms used are - Linear regression, polynomial regression and Decision tree. For Average feedback the classification algorithms used are - Logistic, Decision tree, Naive bayes, SVM and K-nearest neighbour. 

Finally, the optimal operating points are figured out by finding the optimal metric functions values.

The whole process is carried out for one RTX experiment example crowdnav-sequential. This extension can be added for other experiments as well.

* Analysis process embedded to the RTX execution

The analysis process is automated along with the RTX experiments. The process can be stopped by ignoring the relevant codes from the rtxlib\workflow.py file. The relevant code - 
   ```
   info(">start comparison of methods now")
   from compare_methods import regressor_compare_methods, classifier_compare_methods
   classifier_compare_methods()
   regressor_compare_methods()
   ```

The classifier_compare_methods() are for the classification algorithms and regressor_compare_methods() are for the regression algorithms. The cross validation process is executed for each method. The mean values of cross validation for all models are compared with each other to find the optimal model (highest cross validation mean). The model is then further analysed in order to find the optimal operating points. The cross validation code can be shown below - 
   ```
   kfold = model_selection.KFold(n_splits=3, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
   ```

Here, kfold value can be updated by considering different n_splits values. Here we consider only 10 number of data points. So, n_splits = 3 is suitable. But for larger data sets, e.g. for 170 data points, n_splits = 10 - 15 will be suitable. So, the n_splits value can be updated suitably based on the number of data points we are mentioning within the "definition.py" file. "Scoring" is the parameter to mention the process based on which we calculate the cross validation results. For regression we use here mean square error or r score. For classification we use "accuracy" as scoring. 

* Input parameters' values 

The input parameters values can be provided for certain range in order to get a number of operating points among which the most optimal point can be found. The "definition.py" file holds the input parameters values within the dictionary "execution_strategy". The "knobs" array is considered to store the combination inputs variables values.The values can be updated according to the needs. It is expected to provide at least 10 number of input data points to achieve a satisfactory result. The ignore_first_n_results and sample_size are used to controll the sampling of data for each data points combinations.  

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

The output result from RTX experiment is stored into the files results.csv under the corresponding experiment folder (e.g. crowdnav-sequentials). Four columns are stored. First two columns represent two input variables - route_random_sigma and exploration_rate respectively. The last two columns represent two output metrics - Average feedback and Average overhead respectively. The Average feedback consists of discrete / categorized values and the Average overhead consists of continous values. 

```
0.0,0.0,4,2.7262119581425743

0.0,0.1,0,1.9971263285451974

0.0,0.2,0,2.1375472936468833

```

* Individual Machine learning methods

Within the folder machine learning models, all the algorithms are used for the analysis are listed. Each method can be executed individually with the comand - python 'filename'.py. Each model import values from the results.py file. Based on the input and output metric values the model prepares the training and testing data sets. The prediction over the metric values are carried on based on the test dataset. The prediction process for multiple models are compared with each other to decide which method has the most optimized result close to the actual values.

Below are the notes to be considered for these methods. 

1) The files' list is - A) Classification: Logistic regression, KNN, Naive Bayes, Decision Tree, SVM.
                        B) Regression: - Linear regression, Polynomial and Decision Tree.
                        C) Others:- Ttest, outliers, Ttest-onetailed (Showed here only two models- logistic and SVM).

2) For Decision tree, by execution, the .dot files are generated in the same folder. Then those dot files should be translated using query - dot -Tpng F:\RTX-master\RTX-master/tree2.dot -o F:\RTX-master\RTX-master/tree2.png.

3) For ttest, the number of division of rows from results.csv is used very small number (Only 5) as we show only 10 number of data points. It can be increased for more number of result sets.

4) All the models can be embedded with the main RTX execution, only need to update the workflow file and the models' functionality.

* Output for the analysis process

The comparison between the analysis models can be found below - 

![Banner](classification_comparison_result.png)

![Banner](regression_comparison_result.png)


The algorithm models are listed along with their corresponding cv mean values. The first picture shows the classification analysis and the second picture shows the regression analysis. The model with highest mean value is chosen. At the final (2nd picture), the optimal points are also shown.
