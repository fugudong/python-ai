from sklearn.linear_model import *;
import numpy as np
from ReadUserData import *
from LabelProc import get_label_pretty_name
from FeatureProc import get_sensor_names_from_features
import operator
from numpy import *

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names), dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature, is_from_sensor);
        pass;
    X = X[:, use_feature];
    return X;


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train, axis=0);
    std_vec = np.nanstd(X_train, axis=0);
    return (mean_vec, std_vec);


def standardize_features(X, mean_vec, std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1, -1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1, -1));
    X_standard = X_centralized / normalizers;
    return X_standard;


def train_model(X_train, Y_train, M_train, feat_sensor_names, label_names, sensors_to_use, target_label):
    # Project the feature matrix to the features from the desired sensors:
    #print(feat_sensor_names)
    #print(sensors_to_use)
    X_train = project_features_to_selected_sensors(X_train, feat_sensor_names, sensors_to_use);
    #X_train = np.c_[np.ones(X_train.shape[0]), X_train];
    print("== Projected the features to %d features from the sensors: %s" % (
    X_train.shape[1], ', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train, mean_vec, std_vec);
    #print(label_names)
    # The single target label:

    label_ind = label_names.index(target_label)
    y = Y_train[:, label_ind];
    missing_label = M_train[:, label_ind];
    #print(missing_label)
    existing_label = np.logical_not(missing_label);

    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label, :];
    y = y[existing_label];

    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.;

    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y))));

    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.

    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    lr_model = LogisticRegression(class_weight='balanced', solver='liblinear');
    lr_model.fit(X_train, y);
    #print(lr_model.coef_.shape)
    weight = lr_model.coef_;

    # Assemble all the parts of the model:
    model = { \
        'sensors_to_use': sensors_to_use, \
        'target_label': target_label, \
        'mean_vec': mean_vec, \
        'std_vec': std_vec, \
        'lr_model': lr_model,\
        'weight': weight
    };

    return model;

all_weight = [];

def test_model(X_test, Y_test, M_test, target_label, feat_sensor_names, label_names, model):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test, feat_sensor_names, model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (
    X_test.shape[1], ', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test, model['mean_vec'], model['std_vec']);

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;

    label_ind = [];
    for i in range(0, len(target_label)):
        label_ind.append(label_names.index(target_label[i]));

    # Preform the prediction:
    h = [0]*len(Y_test);
    right = float(0);
    for j in range(len(Y_test)):
        voteResult = [0]*len(label_ind);
        for i in range(len(label_ind)):
            h[j] = float(sigmoid(mat(X_test[j]) * mat(all_weight[i])))
            if h[j] > 0.5 and h[j] <= 1:
                voteResult[i] = voteResult[i] + 1 + h[j]
            elif h[j] >= 0 and h[j] <= 0.5:
                voteResult[i] = voteResult[i] - 1 + h[j]
            else:
                print("Properbility wrong");
        h[j] = voteResult.index(max(voteResult))
        if(Y_test[j][label_ind[h[j]]] == True):
            right = right+1;
    print("accuracy: %f" %(right/len(Y_test)));


if __name__ == '__main__':

    sensors_to_use = ['Acc'];
    target_label = ['LYING_DOWN', 'SITTING', 'FIX_walking'];
    uuid = '11B5EC4D-4133-4289-B475-4E737182A406';
    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid);
    #print(feature_names)

    feat_sensor_names = get_sensor_names_from_features(feature_names);
    for i in range(0, len(target_label)):
        model = train_model(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_label[i]);
        all_weight.append(model['weight'].transpose())

    uuid = '00EABED2-271D-49D8-B599-1D4A09240601';
    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid);
    test_model(X, Y, M, target_label, feat_sensor_names, label_names, model);



















