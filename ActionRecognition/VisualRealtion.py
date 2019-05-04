import matplotlib.pyplot as plt;
from LabelProc import get_label_pretty_name
from ReadUserData import *

def figure__feature_scatter_for_labels(X, Y, feature_names, label_names, feature1, feature2, label2color_map):
    feat_ind1 = feature_names.index(feature1);
    feat_ind2 = feature_names.index(feature2);
    example_has_feature1 = np.logical_not(np.isnan(X[:, feat_ind1]));
    example_has_feature2 = np.logical_not(np.isnan(X[:, feat_ind2]));
    example_has_features12 = np.logical_and(example_has_feature1, example_has_feature2);

    fig = plt.figure(figsize=(12, 5), facecolor='white');
    ax1 = plt.subplot(1, 2, 1);
    ax2 = plt.subplot(2, 2, 2);
    ax3 = plt.subplot(2, 2, 4);

    for label in label2color_map.keys():
        label_ind = label_names.index(label);
        pretty_name = get_label_pretty_name(label);
        color = label2color_map[label];
        style = '.%s' % color;

        is_relevant_example = np.logical_and(example_has_features12, Y[:, label_ind]);
        count = sum(is_relevant_example);
        feat1_vals = X[is_relevant_example, feat_ind1];
        feat2_vals = X[is_relevant_example, feat_ind2];
        ax1.plot(feat1_vals, feat2_vals, style, markersize=5, label=pretty_name);

        ax2.hist(X[is_relevant_example, feat_ind1], bins=20, normed=True, color=color, alpha=0.5,
                 label='%s (%d)' % (pretty_name, count));
        ax3.hist(X[is_relevant_example, feat_ind2], bins=20, normed=True, color=color, alpha=0.5,
                 label='%s (%d)' % (pretty_name, count));
        pass;

    ax1.set_xlabel(feature1);
    ax1.set_ylabel(feature2);

    ax2.set_title(feature1);
    ax3.set_title(feature2);

    ax2.legend(loc='best');

    return;

if __name__ == '__main__':
    uuid = '00EABED2-271D-49D8-B599-1D4A09240601';
    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid);
    feature1 = 'proc_gyro:magnitude_stats:time_entropy';  # raw_acc:magnitude_autocorrelation:period';
    feature2 = 'raw_acc:3d:mean_y';
    label2color_map = {'PHONE_IN_HAND': 'b', 'PHONE_ON_TABLE': 'g'};
    figure__feature_scatter_for_labels(X, Y, feature_names, label_names, feature1, feature2, label2color_map);

    feature1 = 'watch_acceleration:magnitude_spectrum:log_energy_band1';
    feature2 = 'watch_acceleration:3d:mean_z';
    label2color_map = {'FIX_walking':'b','WATCHING_TV':'g'};
    figure__feature_scatter_for_labels(X,Y,feature_names,label_names,feature1,feature2,label2color_map);