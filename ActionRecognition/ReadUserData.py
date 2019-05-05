import numpy as np
import gzip
#from StringIO import StringIO
from io import StringIO
import matplotlib as mpl;
import matplotlib.pyplot as plt;

def parse_header_of_csv(csv_str):
    #print(csv_str)
    # Isolate the headline columns:
    csv_str = csv_str.decode()
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:', '');
        pass;

    return (feature_names, label_names);


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO(csv_str.decode()), delimiter=',', skiprows=1);

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int);

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)];

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1];  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat);  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.;  # Y is the label matrix

    return (X, Y, M, timestamps);


'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''


def read_user_data(uuid):
    user_data_file = '%s.features_labels.csv.gz' % uuid;
    print(user_data_file)
    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read();
        pass;

    (feature_names, label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features);

    return (X, Y, M, timestamps, feature_names, label_names);

'''
uuid = '00EABED2-271D-49D8-B599-1D4A09240601';
(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid);
print("The parts of the user's data (and their dimensions):")
print("Every example has its timestamp, indicating the minute when the example was recorded")
print("User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps)))
print ("The primary data files have %d different sensor-features" % len(feature_names))
print ("X is the feature matrix. Each row is an example and each column is a sensor-feature:")
print(X.shape)
print("The primary data files have %s context-labels" % len(label_names))
print("Y is the binary label-matrix. Each row represents an example and each column represents a label.")
print("Value of 1 indicates the label is relevant for the example:")
print(Y.shape)
print("Y is accompanied by the missing-label-matrix, M.")
print("Value of 1 indicates that it is best to consider an entry (example-label pair) as 'missing':")
print(M.shape)
'''

