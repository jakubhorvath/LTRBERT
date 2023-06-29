import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import random
from sklearn.metrics import accuracy_score


def get_presence_count_dict(motif_dict_count, motif_dict_presence, TF_sites):
    for seq in TF_sites:
        for motif in motif_dict_count:
            if len(TF_sites[seq][motif]) > 0:
                motif_dict_count[motif].append(len(TF_sites[seq][motif]))
                motif_dict_presence[motif].append(1)
            else:
                motif_dict_count[motif].append(0)
                motif_dict_presence[motif].append(0)

LTR_TF_sites = pickle.load(open("/var/tmp/xhorvat9_dip/JASPAR/filtered_sequence_motifs.b", "rb"))

non_LTR_TF_sites = {}
for i in range(4):
    d = pickle.load(open(f"/var/tmp/xhorvat9_dip/JASPAR/non_LTR_sequence_motifs{i}.b", "rb"))
    for seq in d:
        non_LTR_TF_sites[seq] = d[seq]


# Process the dictionaries into a dataframe
LTR_motif_dict_count = dict([(key, []) for key in LTR_TF_sites[list(LTR_TF_sites.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in LTR_TF_sites[list(LTR_TF_sites.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, LTR_TF_sites)

non_LTR_motif_dict_count = dict([(key, []) for key in non_LTR_TF_sites[list(non_LTR_TF_sites.keys())[0]]])
non_LTR_motif_dict_presence = dict([(key, []) for key in non_LTR_TF_sites[list(non_LTR_TF_sites.keys())[0]]])
get_presence_count_dict(non_LTR_motif_dict_count, non_LTR_motif_dict_presence, non_LTR_TF_sites)

LTR_motif_count = pd.DataFrame(LTR_motif_dict_count)
nonLTR_motif_count = pd.DataFrame(non_LTR_motif_dict_count)
all_motifs = pd.concat([LTR_motif_count, nonLTR_motif_count])

# create the labels
labels = [1] * len(list(LTR_TF_sites.keys())) + [0] * len(list(non_LTR_TF_sites.keys()))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_motifs, labels, test_size=0.3, random_state=42)

# Create the training pipelines
tfidf_pipeline = Pipeline([("transformer",  TfidfTransformer()),
                           ("classifier", RandomForestClassifier())])

# Set the parameters for the gridsearch
parameters = [
{
	'classifier': (MLPClassifier(solver='adam', activation='logistic', early_stopping=True, validation_fraction=0.1),),
    'classifier__learning_rate_init' : [0.1, 0.05, 0.02, 0.01],
    'classifier__hidden_layer_sizes' : [(10,), (50,), (100,), (200,)],
    'classifier__alpha': [0.0001, 0.001, 0.01],
},
{
	'classifier': (GradientBoostingClassifier(),),
    'classifier__n_estimators' : [50, 100, 200, 400],
    'classifier__learning_rate' : [0.1, 0.05, 0.02, 0.01],
    'classifier__max_depth' : [4, 6, 8],
    'classifier__min_samples_leaf' : [20, 50,100,150]
},
{
	'classifier': (RandomForestClassifier(),),
    'classifier__n_estimators' : [100, 300, 600],
    'classifier__max_depth' : [4, 6, 8, 10, 12],
}]

# Execute gridsearch
grid = GridSearchCV(tfidf_pipeline, parameters, verbose = 10, n_jobs=-1).fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters: ", grid.best_params_)
print("Best Score: ", grid.best_score_)
print("Accuracy: ", accuracy_score(y_test, grid.predict(X_test)))