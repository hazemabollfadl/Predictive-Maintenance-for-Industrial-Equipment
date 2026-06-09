## WTF is going on?
Simply we're trying to predict when an engine gets fucked!
How?
1. We have Training data that are run to failuire -> Feed the Network this Data -> The Network learns what failuire looks like
2. The model currently some patterns of what failuire/Healthy patterns looks like -> test the result on the test_data.
3. Compare the Network test data results with the RUL file containing the actual cycle each engine will failt at.
4. Close reuslts? Good. The results are off? Adjust your Network parameters and compare again.
