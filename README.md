# BSP
This is implementation of "Balancing stability and plasticity in learning topicmodels on short and noisy text streams" (BSP). BSP is a streaming framework built to handling prevailing challenges in streaming environment: sparse, noisy data and plasticity-stability dilemma. Source code contain 3 concepts:
- Holdout_test: Hold out a small set of data for testing and continually learn the training set.
- Chronological_concept: divide the dataset into minibatches based on timestamp. Training a minibatch and then use the next one to test the model
- Drift_forgetting: code for evaluatiing concept drift and catastrophic forgetting phenomenon

# Training
- Holdout_test:
'''
python run.py [dataset_name] [times] [type_model] [rate] [weight] [iters] [start]
'''
- The remaining concepts:
'''
python run.py [times] [type_model] [rate] [weight] [iters] [start]
'''

