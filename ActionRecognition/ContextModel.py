import sklearn.linear_model;
import numpy as np
from ReadUserData import *
from LabelProc import get_label_pretty_name
from FeatureProc import get_sensor_names_from_features

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

    print("== Projected the features to %d features from the sensors: %s" % (
    X_train.shape[1], ', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train, mean_vec, std_vec);
    #print(label_names)

    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:, label_ind];
    #print(y)
    missing_label = M_train[:, label_ind];
    print(missing_label)
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
    lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced');
    lr_model.fit(X_train, y);

    #print(lr_model.class_weight)


    # Assemble all the parts of the model:
    model = { \
        'sensors_to_use': sensors_to_use, \
        'target_label': target_label, \
        'mean_vec': mean_vec, \
        'std_vec': std_vec, \
        'lr_model': lr_model
    };

    return model;


def test_model(X_test, Y_test, M_test, timestamps, feat_sensor_names, label_names, model):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test, feat_sensor_names, model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (
    X_test.shape[1], ', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test, model['mean_vec'], model['std_vec']);

    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:, label_ind];
    missing_label = M_test[:, label_ind];
    existing_label = np.logical_not(missing_label);

    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label, :];
    y = y[existing_label];
    timestamps = timestamps[existing_label];

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;

    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y))));

    # Preform the prediction:

    y_pred = model['lr_model'].predict(X_test);

    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);

    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred, y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y));

    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp + fn);
    specificity = float(tn) / (tn + fp);

    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;

    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp + fp);

    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-" * 10);

    print(
        '* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
    print(
        '** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')

    fig = plt.figure(figsize=(10, 4), facecolor='white');
    ax = plt.subplot(1, 1, 1);
    ax.plot(timestamps[y], 1.4 * np.ones(sum(y)), '|g', markersize=10, label='ground truth');
    ax.plot(timestamps[y_pred], np.ones(sum(y_pred)), '|b', markersize=10, label='prediction');

    seconds_in_day = (60 * 60 * 24);
    tick_seconds = range(timestamps[0], timestamps[-1], seconds_in_day);
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int);

    ax.set_ylim([0.5, 5]);
    ax.set_xticks(tick_seconds);
    ax.set_xticklabels(tick_labels);
    plt.xlabel('days of participation', fontsize=14);
    ax.legend(loc='best');
    plt.title('%s\nGround truth vs. predicted' % get_label_pretty_name(model['target_label']));

    return;

def validate_column_names_are_consistent(old_column_names, new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.");

    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci, old_column_names[ci], new_column_names[ci]));
        pass;
    return;

if __name__ == '__main__':
    sensors_to_use = ['Acc', 'WAcc'];
    target_label = 'FIX_walking';
    uuid = '00EABED2-271D-49D8-B599-1D4A09240601';
    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid);
    #print(feature_names)
    feat_sensor_names = get_sensor_names_from_features(feature_names);
    model = train_model(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_label);
    test_model(X, Y, M, timestamps, feat_sensor_names, label_names, model);

    uuid = '11B5EC4D-4133-4289-B475-4E737182A406';
    (X2, Y2, M2, timestamps2, feature_names2, label_names2) = read_user_data(uuid);

    # All the user data files should have the exact same columns. We can validate it:
    validate_column_names_are_consistent(feature_names, feature_names2);
    validate_column_names_are_consistent(label_names, label_names2);
    test_model(X2, Y2, M2, timestamps2, feat_sensor_names, label_names, model)