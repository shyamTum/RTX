# Simple sequantial run of knob values
name = "CrowdNav-Sequential"

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


def primary_data_reducer(state, newData, wf):
    print("state ",state)
    if(state["count"]==0):
        state["avgFeedback"]=0
    cnt = state["count"]
    state["avg_overhead"] = (state["avg_overhead"] * cnt + newData["overhead"]) / (cnt + 1)
    state["avgFeedback"] = (state["avgFeedback"] * cnt + newData["feedback"]) / (cnt + 1)
    state["count"] = cnt + 1
    # cnt = state["count"]
    # state["avg_overhead"] = (state["avg_overhead"] * cnt + newData["overhead"]) / (cnt + 1)
    # state["count"] = cnt + 1
    # return state
    if(state["avgFeedback"]>3.49):
        state["feedback"] = random.randint(4,5)
    if(3<state["avgFeedback"]<=3.49):
        state["feedback"] = random.randint(3,5)
    if(2.8<state["avgFeedback"]<=3.0):
        # if(feedbackStarProvider(arr)>=4):
        if(newData["feedback"]>=4):
            state["feedback"] = random.randint(4,5)
        # elif(3<=feedbackStarProvider(arr)<4):
        elif(2<newData["feedback"]<4):
            state["feedback"] = random.randint(3,5)
        else:
            state["feedback"] = random.randint(2,4)
    if(state["avgFeedback"]<=2.8):
        if(newData["feedback"]>=4):
            state["feedback"] = 4
        elif(2<=newData["feedback"]<4):
            state["feedback"] = random.randint(3,4)
        else:
            state["feedback"] = random.randint(2,4)
    return state


primary_data_provider = {
    "type": "kafka_consumer",
    "kafka_uri": "localhost:9092",
    "topic": "crowd-nav-trips",
    "serializer": "JSON",
    "data_reducer": primary_data_reducer
}

change_provider = {
    "type": "kafka_producer",
    "kafka_uri": "localhost:9092",
    "topic": "crowd-nav-commands",
    "serializer": "JSON",
}


def evaluator(resultState, wf):
    resultArr=[]
    resultArr.append(resultState["feedback"])
    resultArr.append(resultState["avg_overhead"])
    return str(resultArr).strip('[]').strip('"')
    # return resultState["avg_overhead"]


def state_initializer(state, wf):
    state["count"] = 0
    state["avg_overhead"] = 0
    state["feedback"] = 0
    state["avgFeedback"] = 0
    return state
    # state["count"] = 0
    # state["avg_overhead"] = 0
    # return state
