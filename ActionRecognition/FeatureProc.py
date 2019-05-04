from ReadUserData import *
import matplotlib.pyplot as plt;
from LabelProc import get_label_pretty_name


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc';
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro';
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet';
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc';
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass';
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud';
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP';
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS';
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF';
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat);

        pass;

    return feat_sensor_names;





def figure__feature_track_and_hist(X, feature_names, timestamps, feature_inds):
    seconds_in_day = (60 * 60 * 24);
    days_since_participation = (timestamps - timestamps[0]) / float(seconds_in_day);

    for ind in feature_inds:
        feature = feature_names[ind];
        feat_values = X[:, ind];

        fig = plt.figure(figsize=(10, 3), facecolor='white');

        ax1 = plt.subplot(1, 2, 1);
        ax1.plot(days_since_participation, feat_values, '.-', markersize=3, linewidth=0.1);
        plt.xlabel('days of participation');
        plt.ylabel('feature value');
        plt.title('%d) %s\nfunction of time' % (ind, feature));

        ax1 = plt.subplot(1, 2, 2);
        existing_feature = np.logical_not(np.isnan(feat_values));
        ax1.hist(feat_values[existing_feature], bins=30);
        plt.xlabel('feature value');
        plt.ylabel('count');
        plt.title('%d) %s\nhistogram' % (ind, feature));

        pass;
    return;


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
    feat_sensor_names = get_sensor_names_from_features(feature_names);
    for (fi, feature) in enumerate(feature_names):
        print("%3d) %s %s" % (fi, feat_sensor_names[fi].ljust(10), feature));
        pass;
    feature_inds = [0, 102, 133, 148, 157, 158];
    figure__feature_track_and_hist(X, feature_names, timestamps, feature_inds);

    print("The phone-state (PS) features are represented as binary indicators:");
    feature_inds = [205, 223];
    figure__feature_track_and_hist(X, feature_names, timestamps, feature_inds);

    feature1 = 'proc_gyro:magnitude_stats:time_entropy';  # raw_acc:magnitude_autocorrelation:period';
    feature2 = 'raw_acc:3d:mean_y';
    label2color_map = {'PHONE_IN_HAND': 'b', 'PHONE_ON_TABLE': 'g'};
    figure__feature_scatter_for_labels(X, Y, feature_names, label_names, feature1, feature2, label2color_map);

    feature1 = 'watch_acceleration:magnitude_spectrum:log_energy_band1';
    feature2 = 'watch_acceleration:3d:mean_z';
    label2color_map = {'FIX_walking': 'b', 'WATCHING_TV': 'g'};
    figure__feature_scatter_for_labels(X, Y, feature_names, label_names, feature1, feature2, label2color_map);