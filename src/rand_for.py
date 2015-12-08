from sklearn.ensemble import RandomForestClassifier
import numpy as np
from feature_set import get_all_feature_sets



setts = get_all_feature_sets()

keys = []
scores = []

for key in setts:

    keys.append(key)

    print("Loading %sand associated labels..." % key)

    examples = setts[key]['train_X']
    labels = setts[key]['train_Y']
    test_examples = setts[key]['test_X']
    

    test_labels = setts[key]['test_Y']
    Forest = RandomForestClassifier(100)

    Forest.fit_transform(examples, labels)

    print Forest.score(test_examples, test_labels)

    scores.append(Forest.score(test_examples, test_labels))



with open("../data/random_forest_all_feature_combinations.txt", "wb") as f:
    for i in range(len(keys)):
        f.write("%s, %d\n" %(keys[i], scores[i]))

