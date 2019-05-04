import matplotlib.pyplot as plt;
import numpy as np
from ReadUserData import *

def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'

    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;

    label = label.replace('__', ' (').replace('_', ' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m', 'I\'m');
    return label;



def figure__pie_chart(Y, label_names, labels_to_display, title_str, ax):
    portion_of_time = np.mean(Y, axis=0);
    portions_to_display = [portion_of_time[label_names.index(label)] for label in labels_to_display];
    pretty_labels_to_display = [get_label_pretty_name(label) for label in labels_to_display];

    ax.pie(portions_to_display, labels=pretty_labels_to_display, autopct='%.2f%%');
    ax.axis('equal');
    plt.title(title_str);
    return;


def get_actual_date_labels(tick_seconds):
    import datetime;
    import pytz;

    time_zone = pytz.timezone('US/Pacific');  # Assuming the data comes from PST time zone
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    datetime_labels = [];
    for timestamp in tick_seconds:
        tick_datetime = datetime.datetime.fromtimestamp(timestamp, tz=time_zone);
        weekday_str = weekday_names[tick_datetime.weekday()];
        time_of_day = tick_datetime.strftime('%I:%M%p');
        datetime_labels.append('%s\n%s' % (weekday_str, time_of_day));
        pass;

    return datetime_labels;


def figure__context_over_participation_time(timestamps, Y, label_names, labels_to_display, label_colors,
                                            use_actual_dates=False):
    fig = plt.figure(figsize=(10, 7), facecolor='white');
    ax = plt.subplot(1, 1, 1);

    seconds_in_day = (60 * 60 * 24);

    ylabels = [];
    ax.plot(timestamps, len(ylabels) * np.ones(len(timestamps)), '|', color='0.5', label='(Collected data)');
    ylabels.append('(Collected data)');

    for (li, label) in enumerate(labels_to_display):
        lind = label_names.index(label);
        is_label_on = Y[:, lind];
        label_times = timestamps[is_label_on];

        label_str = get_label_pretty_name(label);
        ax.plot(label_times, len(ylabels) * np.ones(len(label_times)), '|', color=label_colors[li], label=label_str);
        ylabels.append(label_str);
        pass;

    tick_seconds = range(timestamps[0], timestamps[-1], seconds_in_day);
    if use_actual_dates:
        tick_labels = get_actual_date_labels(tick_seconds);
        plt.xlabel('Time in San Diego', fontsize=14);
        pass;
    else:
        tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int);
        plt.xlabel('days of participation', fontsize=14);
        pass;

    ax.set_xticks(tick_seconds);
    ax.set_xticklabels(tick_labels, fontsize=14);

    ax.set_yticks(range(len(ylabels)));
    ax.set_yticklabels(ylabels, fontsize=14);

    ax.set_ylim([-1, len(ylabels)]);
    ax.set_xlim([min(timestamps), max(timestamps)]);

    return;


def jaccard_similarity_for_label_pairs(Y):
    (n_examples,n_labels) = Y.shape;
    Y = Y.astype(int);
    # For each label pair, count cases of:
    # Intersection (co-occurrences) - cases when both labels apply:
    both_labels_counts = np.dot(Y.T,Y);
    # Cases where neither of the two labels applies:
    neither_label_counts = np.dot((1-Y).T,(1-Y));
    # Union - cases where either of the two labels (or both) applies (this is complement of the 'neither' cases):
    either_label_counts = n_examples - neither_label_counts;
    # Jaccard similarity - intersection over union:
    J = np.where(either_label_counts > 0, both_labels_counts.astype(float) / either_label_counts, 0.);
    return J;


if __name__ == '__main__':
    uuid = '1155FF54-63D3-4AB2-9863-8385D0BD0A13';
    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid);
    n_examples_per_label = np.sum(Y,axis=0);
    labels_and_counts = zip(label_names,n_examples_per_label);
    sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1]);
    print("How many examples does this user have for each contex-label:")
    print("-"*20)

    for (label,count) in sorted_labels_and_counts:
        print("label %s - %d minutes" % (label,count))
        pass;
    print("Since the collection of labels relied on self-reporting in-the-wild, the labeling may be incomplete.");
    print("For instance, the users did not always report the position of the phone.");
    fig = plt.figure(figsize=(15, 5), facecolor='white');

    ax1 = plt.subplot(1, 2, 1);
    labels_to_display = ['LYING_DOWN', 'SITTING', 'OR_standing', 'FIX_walking', 'FIX_running'];
    figure__pie_chart(Y, label_names, labels_to_display, 'Body state', ax1);

    ax2 = plt.subplot(1, 2, 2);
    labels_to_display = ['PHONE_IN_HAND', 'PHONE_IN_BAG', 'PHONE_IN_POCKET', 'PHONE_ON_TABLE'];
    figure__pie_chart(Y, label_names, labels_to_display, 'Phone position', ax2);

    print("Here is a track of when the user was engaged in different contexts.");
    print(
        "The bottom row (gray) states when sensors were recorded (the data-collection app was not running all the time).");
    print("The context-labels annotations were self-reported by ther user (and then cleaned by the researchers).")

    labels_to_display = ['LYING_DOWN', 'LOC_home', 'LOC_main_workplace', 'SITTING', 'OR_standing', 'FIX_walking', \
                         'IN_A_CAR', 'ON_A_BUS', 'EATING'];
    label_colors = ['g', 'y', 'b', 'c', 'm', 'b', 'r', 'k', 'purple'];
    figure__context_over_participation_time(timestamps, Y, label_names, labels_to_display, label_colors);
    J = jaccard_similarity_for_label_pairs(Y);

    print("Label-pairs with higher color value tend to occur together more.");

    fig = plt.figure(figsize=(10, 10), facecolor='white');
    ax = plt.subplot(1, 1, 1);
    plt.imshow(J, interpolation='none');
    plt.colorbar();

    pretty_label_names = [get_label_pretty_name(label) for label in label_names];
    n_labels = len(label_names);
    ax.set_xticks(range(n_labels));
    ax.set_xticklabels(pretty_label_names, rotation=45, ha='right', fontsize=7);
    ax.set_yticks(range(n_labels));
    ax.set_yticklabels(pretty_label_names, fontsize=7);
