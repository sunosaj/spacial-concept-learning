import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.stats import kde
import pandas as pd
from gensim.models import KeyedVectors
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import math
import scipy.stats as stats
import pylab as pl
import seaborn as sns


def get_pred_data(pred):
    file_address = 'new_pred/data_pred_' + pred + '.csv'
    data = pd.read_csv(file_address, encoding="ISO-8859-1")
    return data


def get_most_dominant_pred(table_name):
    # Retrieve counts for each pred occurrence for each item in table
    table_location = "pred_occurrences/" + table_name + ".p"
    table = pickle.load(open(table_location, "rb"))

    most_dominant_pred = []

    # Loop through data rows and add index of largest integer for each row (most dominant predicate for this row)
    for i in range(len(table)):
        most_dominant_pred.append(np.argmax(table[i]))

    return most_dominant_pred


def plot_normalized_centre_points(pred):
    # Get obj and subj centre points and normalize diffs
    x_normalized_diff = []
    y_normalized_diff = []

    # Weird images
    outlier_relationships = []

    data = get_pred_data(pred)

    for i in range(len(data)):
        obj_centre_x = float(data.iloc[i, 5]) + (float(data.iloc[i, 4]) / 2.0)
        obj_centre_y = float(data.iloc[i, 6]) + (float(data.iloc[i, 3]) / 2.0)
        subj_centre_x = float(data.iloc[i, 12]) + (float(data.iloc[i, 11]) / 2.0)
        subj_centre_y = float(data.iloc[i, 13]) + (float(data.iloc[i, 10]) / 2.0)

        x_normalized_diff.append((subj_centre_x - obj_centre_x) / float(data.iloc[i, 15]))
        y_normalized_diff.append((obj_centre_y - subj_centre_y) / float(data.iloc[i, 16]))

        # For specific predicates #
        if pred == 'above':
            if y_normalized_diff[-1] < 0:
                outlier_relationships.append(data.iloc[i, ])
        elif pred == 'below':
            if y_normalized_diff[-1] > 0:
                outlier_relationships.append(data.iloc[i, ])
        ###########################

    # Graph the normalized diff coordinates
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.title(pred)
    plt.plot(x_normalized_diff, y_normalized_diff, 'ro', markersize=1)
    plt.axis([-1, 1, -1, 1])

    # Add x, y axis
    plt.axhline(linewidth=0.5, color='k')
    plt.axvline(linewidth=0.5, color='k')

    # Keep plot graph square
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.show()

    # Kernel Density Estimator
    plt.subplot(1, 2, 2)
    plt.axis([-1, 1, -1, 1])

    kde_x_normalized_diff = np.asarray(x_normalized_diff)
    kde_y_normalized_diff = np.asarray(y_normalized_diff)

    k = kde.gaussian_kde([kde_x_normalized_diff, kde_y_normalized_diff])
    xi, yi = np.mgrid[kde_x_normalized_diff.min():kde_x_normalized_diff.max():100j,
                      kde_y_normalized_diff.min():kde_y_normalized_diff.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_earth_r)
    plt.axhline(linewidth=0.5, color='k')
    plt.axvline(linewidth=0.5, color='k')
    plt.gca().set_aspect('equal', adjustable='box')
    ###########################

    plt.title(pred)
    plt.colorbar()

    # Save plots
    plt.savefig("new_pred_plot/" + pred + "_plot.png", dpi=300)

    # plt.show()

    return outlier_relationships


def plot_all(predicates):

    # plt.figure(figsize=(14, 5))
    # plt.subplot(1, 2, 1)
    plt.title("All")
    plt.axis([-1, 1, -1, 1])
    # Add x, y axis
    plt.axhline(linewidth=0.5, color='k')
    plt.axvline(linewidth=0.5, color='k')

    # Keep plot graph square
    plt.gca().set_aspect('equal', adjustable='box')
    for pred in predicates:
        # Get obj and subj centre points and normalize diffs
        x_normalized_diff = []
        y_normalized_diff = []

        data = get_pred_data(pred)

        for i in range(len(data)):
            obj_centre_x = float(data.iloc[i, 5]) + (float(data.iloc[i, 4]) / 2.0)
            obj_centre_y = float(data.iloc[i, 6]) + (float(data.iloc[i, 3]) / 2.0)
            subj_centre_x = float(data.iloc[i, 12]) + (float(data.iloc[i, 11]) / 2.0)
            subj_centre_y = float(data.iloc[i, 13]) + (float(data.iloc[i, 10]) / 2.0)

            x_normalized_diff.append((subj_centre_x - obj_centre_x) / float(data.iloc[i, 15]))
            y_normalized_diff.append((obj_centre_y - subj_centre_y) / float(data.iloc[i, 16]))

        # Graph the normalized diff coordinates
        plt.plot(x_normalized_diff, y_normalized_diff, 'o', markersize=1)

        print(pred, "done")

    # # Save plot
    # plt.savefig("new_pred_plot/combined_one_plot.pdf", dpi=300)

    plt.show()


def box_images(predicates):
    # # Box specific images
    # pred = 'on'
    # i = 1473
    #
    # data = get_pred_data(pred)
    # # print(max(data.iloc[i, 3] * data.iloc[i, 4], data.iloc[i, 10] * data.iloc[i, 11]))
    # image_address = 'images/' + str(data.iloc[i, 0]) + '.jpg'
    # im = np.array(Image.open(image_address), dtype=np.uint8)
    #
    # # Create figure and axes
    # fig, ax = plt.subplots(1)
    #
    # # Display the image
    # ax.imshow(im)
    #
    # # Create a Rectangle patch
    # obj_box = patches.Rectangle((int(data.iloc[i, 5]), int(data.iloc[i, 6])), int(data.iloc[i, 4]),
    #                             int(data.iloc[i, 3]), linewidth=1, edgecolor='b', facecolor='none')
    # subj_box = patches.Rectangle((int(data.iloc[i, 12]), int(data.iloc[i, 13])), int(data.iloc[i, 11]),
    #                              int(data.iloc[i, 10]), linewidth=1, edgecolor='r', facecolor='none')
    #
    # # Add the patch to the Axes
    # ax.add_patch(obj_box)
    # ax.add_patch(subj_box)
    #
    # # # Add centroids
    # # plt.plot(int(data.iloc[i, 5]) + (0.5 * int(data.iloc[i, 4])),
    # #          int(data.iloc[i, 6]) + (0.5 * int(data.iloc[i, 3])), 'bo', markersize=3)
    # # plt.plot(int(data.iloc[i, 12]) + (0.5 * int(data.iloc[i, 11])),
    # #          int(data.iloc[i, 13]) + (0.5 * int(data.iloc[i, 10])), 'ro', markersize=3)
    #
    #
    # # Label obj and subj
    # plt.annotate("Object: " + data.iloc[i, 2], (int(data.iloc[i, 5]), int(data.iloc[i, 6])), color='k')
    # plt.annotate("Subject: " + data.iloc[i, 9], (int(data.iloc[i, 12]), int(data.iloc[i, 13])), color='k')
    #
    # plt.title('Preposition: ' + pred)
    # # plt.title(pred + ": " + str(i))
    # plt.show()

    # Box all predicates
    for pred in predicates:
        data = get_pred_data(pred)
        for i in range(332, len(data)):
            # Decide whether row i fulfills rules #
            factor = math.sqrt(0.5 / (int(data.iloc[i, 3] * int(data.iloc[i, 4]))))

            # get max y
            y_diff_1 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - data.iloc[i, 13])
            y_diff_2 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - (data.iloc[i, 13] + data.iloc[i, 10]))
            max_y = max(y_diff_1, y_diff_2, 0.5 * data.iloc[i, 3])

            # get max x
            x_diff_1 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - data.iloc[i, 12])
            x_diff_2 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - (data.iloc[i, 12] + data.iloc[i, 11]))
            max_x = max(x_diff_1, x_diff_2, 0.5 * data.iloc[i, 4])

            # max separation
            longest_separation = max(max_y * factor, max_x * factor)

            #######################################
            if fulfills_rules(data.iloc[i,], longest_separation, factor):
                image_address = 'images/' + str(data.iloc[i, 0]) + '.jpg'
                im = np.array(Image.open(image_address), dtype=np.uint8)

                # Create figure and axes
                fig, ax = plt.subplots(1)

                # Display the image
                ax.imshow(im)

                # Create a Rectangle patch
                obj_box = patches.Rectangle((int(data.iloc[i, 5]), int(data.iloc[i, 6])), int(data.iloc[i, 4]),
                                            int(data.iloc[i, 3]), linewidth=1, edgecolor='b', facecolor='none')
                subj_box = patches.Rectangle((int(data.iloc[i, 12]), int(data.iloc[i, 13])), int(data.iloc[i, 11]),
                                             int(data.iloc[i, 10]), linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(obj_box)
                ax.add_patch(subj_box)

                # Label obj and subj
                plt.annotate("Object: " + data.iloc[i, 2], (int(data.iloc[i, 5]), int(data.iloc[i, 6])), color='k')
                plt.annotate("Subject: " + data.iloc[i, 9], (int(data.iloc[i, 12]), int(data.iloc[i, 13])), color='w')

                plt.title("Preposition: " + pred)
                # plt.title(pred + ": " + str(i))
                print(str(i))
                plt.show()


def word_to_vector(table_name, pred_order_name, model, vocab):
    # Retrieve labels for each row in table
    label_location = "pred_occurrences/" + table_name + ".p"
    words = pickle.load(open(label_location, "rb"))

    pred_order_location = "pred_occurrences/" + pred_order_name + ".p"
    preds = pickle.load(open(pred_order_location, "rb"))

    word_vectors = []
    new_pred_order = []
    for i in range(len(words)):
        # if (subj, obj)
        if isinstance(words[i], tuple):
            if words[i][0] in vocab and words[i][1] in vocab:
                word_vectors.append(model[words[i][0]] - model[words[i][1]])
                new_pred_order.append(preds[i])
        # if subj, or obj
        else:
            if words[i] in vocab:
                word_vectors.append(model[words[i]])
                new_pred_order.append(preds[i])

    return word_vectors, new_pred_order


def pickle_file(table, name):
    # https://wiki.python.org/moin/UsingPickle
    file_name = "pred_occurrences/" + name + ".p"
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(table, fp)


def reduce_dimensionality(vector_names, dominant_pred, reduction_method):
    # Retrieve vectors for each word
    label_location = "pred_occurrences/" + vector_names + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve predicates associated with each word only if reduction_method is 'lda'
    vector_predicates = None
    if dominant_pred is not None:
        pred_order_location = "pred_occurrences/" + dominant_pred + ".p"
        vector_predicates = np.asarray(pickle.load(open(pred_order_location, "rb")))

    # print('len(vectors)', len(vectors))
    # print('len(vector_predicates)', len(vector_predicates))

    # Number of dimensions to reduce to
    n_dimensions = 43
    print('n_dimensions', n_dimensions)

    if reduction_method == 'pca':
        pca = PCA(n_components=n_dimensions)
        pca.fit(vectors)
        new_vectors = pca.transform(vectors)

    elif reduction_method == 'tsne':
        vectors_embedded = TSNE(n_components=3).fit_transform(vectors)
        new_vectors = vectors_embedded

    elif reduction_method == 'lda':
        # pick another solver to set shrinkage = 0.5, and set n_components=k-1
        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.5, n_components=n_dimensions)
        clf.fit(vectors, vector_predicates)
        new_vectors = clf.transform(vectors)
        print('lda new_vectors', new_vectors.shape)

    return new_vectors


def plot_vectors(vectors_file_name, vector_preds_file_name, title, n_dimensions, predicates):
    # Retrieve dimensionally reduced vectors for each subj/obj/subj-obj
    label_location = "pred_occurrences/" + vectors_file_name + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve corresponding most dominant predicate for each subj/obj/subj-obj
    label_location = "pred_occurrences/" + vector_preds_file_name + ".p"
    vector_preds = np.asarray(pickle.load(open(label_location, "rb")))

    print('hi1')

    # Colors for each predicate
    # pred_to_color = {1: [predicates[1], 'black'], 11: [predicates[11], 'green'], 17: [predicates[17], 'violet'],
    #                  20: [predicates[20], 'blue'], 23: [predicates[23], 'orange'], 25: [predicates[25], 'red'],
    #                  37: [predicates[38], 'indigo'], 41: [predicates[41], 'yellow']}

    # print(predicates)

    # # Colors for each predicate (reduced list)
    # pred_to_color = {0: [predicates[0], 'blue'], 1: [predicates[1], 'red']}
    #
    # preds = []
    # colors = []
    # for i in range(len(predicates)):
    #     if i in pred_to_color:
    #         preds.append(pred_to_color[i][0])
    #         colors.append(pred_to_color[i][1])
    #
    # print('preds', preds)
    # print('colors', colors)
    #
    # handles_patches = [mpatches.Patch(color=colors[i], label="{:s}".format(preds[i])) for i in range(len(preds))]
    #
    # print(vectors.shape)

    # Plot 2-d
    if n_dimensions == '2':
        print('hi2')
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        # plt.legend(handles=handles_patches, ncol=2)

        kde_x = []
        kde_y = []

        # for i in range(len(vectors)):
        #     if vector_preds[i] == 20 or vector_preds[i] == 25:
        #         if vector_preds[i] == 20:
        #             plt.plot(vectors[i, 0], vectors[i, 1], 'bo', markersize=1)
        #         elif vector_preds[i] == 25:
        #             plt.plot(vectors[i, 0], vectors[i, 1], 'ro', markersize=1)
        #         kde_x.append(vectors[i, 0])
        #         kde_y.append((vectors[i, 1]))

        # IN
        plt.plot(vectors[114974:340327, 0], vectors[114974:340327, 1], 'bo', markersize=1)
        kde_x.append(vectors[114974:340327, 0])
        kde_y.append((vectors[114974:340327, 1]))

        # ON
        plt.plot(vectors[373986:776315, 0], vectors[373986:776315, 1], 'ro', markersize=1)
        kde_x.append(vectors[373986:776315, 0])
        kde_y.append((vectors[373986:776315, 1]))

        print('hi2')
        plt.title(title + ' 2d')
        # plt.axis([-1000, 1000, -1000, 1000])
        plt.gca().set_aspect('equal', adjustable='box')

        # Add x, y axis
        plt.axhline(linewidth=0.5, color='k')
        plt.axvline(linewidth=0.5, color='k')

        # Legend
        blue_patch = mpatches.Patch(color='blue', label='in')
        red_patch = mpatches.Patch(color='red', label='on')
        plt.legend(handles=[blue_patch, red_patch])

        # plt.savefig("word_vector_plot/" + vectors_file_name + "_plot_2d.pdf")

        # # Kernel Density Estimator #
        # plt.subplot(1, 2, 2)
        # plt.axis([-2, 2, -2, 2])
        #
        # kde_x_normalized_diff = np.asarray(kde_x)
        # kde_y_normalized_diff = np.asarray(kde_y)
        #
        # k = kde.gaussian_kde([kde_x_normalized_diff, kde_y_normalized_diff])
        # xi, yi = np.mgrid[kde_x_normalized_diff.min():kde_x_normalized_diff.max():100j,
        #          kde_y_normalized_diff.min():kde_y_normalized_diff.max():100j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        #
        # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_earth_r)
        # plt.axhline(linewidth=0.5, color='k')
        # plt.axvline(linewidth=0.5, color='k')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.colorbar()
        ############################

        plt.show()

    # # Plot 3-d
    # elif n_dimensions == '3':
    #     # plt.plot(vectors[:3, 0], vectors[:3, 1], 'ro', markersize=1)
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     plt.legend(handles=handles_patches, ncol=2)
    #
    #     for i in range(len(vectors)):
    #         # print(len(vectors))
    #         if vector_preds[i] in pred_to_color:
    #             plt.plot([vectors[i, 0]], [vectors[i, 1]], [vectors[i, 2]], marker='o', markersize=1,
    #                      color=pred_to_color[vector_preds[i]][1])
    #
    #     # ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=0.1, c='b')
    #     plt.title(title + ' 3d')
    #     plt.savefig("word_vector_plot/" + vectors_file_name + "_plot_3d.pdf")


def find_longest_separation(predicates):
    # might also want to look at which image contains longest_edge and longest_separation
    # so get image id too when u update max

    # longest_separation = 0
    # pred = 'in'
    # i = 179
    #
    # data = get_pred_data(pred)
    #
    # # get longest edge
    # old_longest_edge = longest_edge
    # longest_edge = max(longest_edge, data.iloc[i, 3], data.iloc[i, 4], data.iloc[i, 10], data.iloc[i, 11])
    # if longest_edge > old_longest_edge:
    #     longest_edge_pred = pred
    #     longest_edge_pred_index = i
    #     longest_edge_img_id = data.iloc[i, 0]
    #
    # # get max y
    # y_diff_1 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - data.iloc[i, 13])
    # print('y_diff_1', y_diff_1)
    # y_diff_2 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - (data.iloc[i, 13] + data.iloc[i, 10]))
    # print('y_diff_2', y_diff_2)
    # max_y = max(y_diff_1, y_diff_2, 0.5 * data.iloc[i, 3])
    # print('max_y', max_y)
    #
    # # get max x
    # x_diff_1 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - data.iloc[i, 12])
    # print('x_diff_1', x_diff_1)
    # x_diff_2 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - (data.iloc[i, 12] + data.iloc[i, 11]))
    # print('x_diff_2', x_diff_2)
    # max_x = max(x_diff_1, x_diff_2, 0.5 * data.iloc[i, 4])
    # print('max_x', max_x)
    #
    # # max separation
    # old_longest_separation = longest_separation
    # longest_separation = max(longest_separation, max_y, max_x)
    # if longest_separation > old_longest_separation:
    #     longest_separation_pred = pred
    #     longest_separation_pred_index = i
    #     longest_separation_img_id = data.iloc[i, 0]
    #
    # print(pred, 'done')
    #
    # return longest_edge, longest_separation, longest_edge_pred, longest_edge_pred_index, longest_edge_img_id, \
    #        longest_separation_pred, longest_separation_pred_index, longest_separation_img_id

    longest_separation = 0
    # longest_separation_pred = predicates[0]
    # longest_separation_pred_index = 0
    # longest_separation_img_id = 0

    total_lines_of_data = 0
    total_used_lines_of_data = 0
    for pred in predicates:
        data = get_pred_data(pred)
        pred_lines_of_data = len(data)
        pred_used_lines_of_data = 0
        total_lines_of_data += pred_lines_of_data
        for i in range(len(data)):
            # Proceed with this line of data only if it fulfills certain rules

            factor = math.sqrt(0.5 / (int(data.iloc[i, 3]) * int(data.iloc[i, 4])))

            # get max y
            y_diff_1 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - data.iloc[i, 13])
            y_diff_2 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - (data.iloc[i, 13] + data.iloc[i, 10]))
            max_y = max(y_diff_1, y_diff_2, 0.5 * data.iloc[i, 3])

            # get max x
            x_diff_1 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - data.iloc[i, 12])
            x_diff_2 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - (data.iloc[i, 12] + data.iloc[i, 11]))
            max_x = max(x_diff_1, x_diff_2, 0.5 * data.iloc[i, 4])

            # max separation
            old_longest_separation = longest_separation
            new_longest_separation = max(longest_separation, max_y * factor, max_x * factor)
            if fulfills_rules(data.iloc[i, ], new_longest_separation, factor):
                if new_longest_separation > old_longest_separation:
                    longest_separation = new_longest_separation
                    longest_separation_pred = pred
                    longest_separation_pred_index = i
                    longest_separation_img_id = data.iloc[i, 0]
                pred_used_lines_of_data += 1

        total_used_lines_of_data += pred_used_lines_of_data

        print('pred_lines_of_data', pred_lines_of_data)
        print('pred_used_lines_of_data', pred_used_lines_of_data)
        print('percent used', pred_used_lines_of_data / pred_lines_of_data)
        print(pred, 'done')

    print('total_lines_of_data', total_lines_of_data)
    print('used_lines_of_data', total_used_lines_of_data)
    print('percent used', total_used_lines_of_data / total_lines_of_data)

    return longest_separation, longest_separation_pred, longest_separation_pred_index, longest_separation_img_id


def get_bounding_box_vectors(pred, decrease_factor, plot_size):
    # # Get bounding box of specific image for specific predicate
    # pred = 'on'
    # i = 35610
    #
    # data = get_pred_data(pred)
    #
    # # create white plot of -normalized_longest_separation to +normalized_longest_separation
    # fig, ax = plt.subplots(1)
    # plt.axis([-plot_size, plot_size, -plot_size, plot_size])
    #
    # # plot object bounding box at centre
    # # https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html
    # # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    # obj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * decrease_factor,
    #                              0 - 0.5 * int(data.iloc[i, 3]) * decrease_factor),
    #                             int(data.iloc[i, 4]) * decrease_factor, int(data.iloc[i, 3]) * decrease_factor,
    #                             linewidth=1, edgecolor='b', facecolor='none')
    #
    # # plot subject relative to object
    # subj_x_relative = (int(data.iloc[i, 12]) - int(data.iloc[i, 5])) * decrease_factor
    # subj_y_relative = ((int(data.iloc[i, 6]) + int(data.iloc[i, 3])) -
    #                    (int(data.iloc[i, 13]) + int(data.iloc[i, 10]))) * decrease_factor
    # subj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * decrease_factor + subj_x_relative,
    #                              0 - 0.5 * int(data.iloc[i, 3]) * decrease_factor + subj_y_relative),
    #                              int(data.iloc[i, 11]) * decrease_factor, int(data.iloc[i, 10]) * decrease_factor,
    #                              linewidth=1, edgecolor='r', facecolor='none')
    #
    # ax.add_patch(obj_box)
    # ax.add_patch(subj_box)
    #
    # # Keep plot graph square
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(pred + ": " + str(data.iloc[i, 0]))
    # plt.show()

    # Get bounding boxes for all images of given predicate
    data = get_pred_data(pred)

    for i in range(len(data)):
        # create white plot of -normalized_longest_separation to +normalized_longest_separation
        fig, ax = plt.subplots(1)
        plt.axis([-plot_size, plot_size, -plot_size, plot_size])

        # plot object bounding box at centre
        # https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html
        # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
        obj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * decrease_factor,
                                     0 - 0.5 * int(data.iloc[i, 3]) * decrease_factor),
                                    int(data.iloc[i, 4]) * decrease_factor, int(data.iloc[i, 3]) * decrease_factor,
                                    linewidth=1, edgecolor='b', facecolor='none')

        # plot subject relative to object
        subj_x_relative = (int(data.iloc[i, 12]) - int(data.iloc[i, 5])) * decrease_factor
        subj_y_relative = ((int(data.iloc[i, 6]) + int(data.iloc[i, 3])) -
                           (int(data.iloc[i, 13]) + int(data.iloc[i, 10]))) * decrease_factor
        subj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * decrease_factor + subj_x_relative,
                                     0 - 0.5 * int(data.iloc[i, 3]) * decrease_factor + subj_y_relative),
                                     int(data.iloc[i, 11]) * decrease_factor, int(data.iloc[i, 10]) * decrease_factor,
                                     linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(obj_box)
        ax.add_patch(subj_box)

        # Keep plot graph square
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(pred + ": " + str(i))
        plt.show()

        # Turn plot into one long 300-dimension 1d array by __

        # Return these arrays, as well as their corresponding predicate for LDA later


def get_normalized_bounding_box_vectors(pred, plot_size):
    # Get bounding box of specific image for specific predicate
    pred = 'above'
    i = 53

    data = get_pred_data(pred)

    # create white plot of -normalized_longest_separation to +normalized_longest_separation
    fig, ax = plt.subplots(1)
    plt.axis([-plot_size, plot_size, -plot_size, plot_size])

    factor = math.sqrt(0.5 / (int(data.iloc[i, 3] * int(data.iloc[i, 4]))))

    # plot object bounding box at centre
    # https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html
    # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    obj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * factor,
                                 0 - 0.5 * int(data.iloc[i, 3]) * factor),
                                int(data.iloc[i, 4]) * factor, int(data.iloc[i, 3]) * factor,
                                linewidth=1, edgecolor='b', facecolor='none')

    # plot subject relative to object
    subj_x_relative = (int(data.iloc[i, 12]) - int(data.iloc[i, 5])) * factor
    subj_y_relative = ((int(data.iloc[i, 6]) + int(data.iloc[i, 3])) -
                       (int(data.iloc[i, 13]) + int(data.iloc[i, 10]))) * factor
    subj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * factor + subj_x_relative,
                                 0 - 0.5 * int(data.iloc[i, 3]) * factor + subj_y_relative),
                                 int(data.iloc[i, 11]) * factor, int(data.iloc[i, 10]) * factor,
                                 linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(obj_box)
    ax.add_patch(subj_box)

    # Keep plot graph square
    plt.gca().set_aspect('equal', adjustable='box')

    ax = fig.gca()
    ax.set_xticks(np.arange(-1, 1, 0.01))
    ax.set_yticks(np.arange(-1, 1., 0.01))
    plt.grid(True)
    plt.rc('grid', linestyle="-", color='black')
    plt.xlabel('200')
    plt.ylabel('200')
    # plt.axis('off')

    # plt.title(pred + ": " + str(i))
    plt.title(pred)
    plt.savefig("../Presentation-Images/" + pred + "_" + str(i)+ "_bb_plot_grid.png", dpi=2400)
    plt.show()

    # # Get bounding boxes for all images of given predicate
    # data = get_pred_data(pred)
    #
    # # fig, ax = plt.subplots(1)
    # # plt.axis([-plot_size, plot_size, -plot_size, plot_size])
    #
    # for i in range(len(data)):
    #     # create white plot of -normalized_longest_separation to +normalized_longest_separation
    #     # fig, ax = plt.subplots(1)
    #     # plt.axis([-plot_size, plot_size, -plot_size, plot_size])
    #
    #     # Decide whether row i fulfills rules #
    #     factor = math.sqrt(0.5 / (int(data.iloc[i, 3] * int(data.iloc[i, 4]))))
    #
    #     # get max y
    #     y_diff_1 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - data.iloc[i, 13])
    #     y_diff_2 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - (data.iloc[i, 13] + data.iloc[i, 10]))
    #     max_y = max(y_diff_1, y_diff_2, 0.5 * data.iloc[i, 3])
    #
    #     # get max x
    #     x_diff_1 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - data.iloc[i, 12])
    #     x_diff_2 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - (data.iloc[i, 12] + data.iloc[i, 11]))
    #     max_x = max(x_diff_1, x_diff_2, 0.5 * data.iloc[i, 4])
    #
    #     # max separation
    #     longest_separation = max(max_y * factor, max_x * factor)
    #
    #     #######################################
    #     if fulfills_rules(data.iloc[i, ], longest_separation, factor):
    #         # create white plot of -normalized_longest_separation to +normalized_longest_separation
    #         fig, ax = plt.subplots(1)
    #         plt.axis([-plot_size, plot_size, -plot_size, plot_size])
    #
    #         # plot object bounding box at centre
    #         # https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html
    #         # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    #         obj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * factor,
    #                                      0 - 0.5 * int(data.iloc[i, 3]) * factor),
    #                                     int(data.iloc[i, 4]) * factor, int(data.iloc[i, 3]) * factor,
    #                                     linewidth=1, edgecolor='b', facecolor='none')
    #
    #         # plot subject relative to object
    #         subj_x_relative = (int(data.iloc[i, 12]) - int(data.iloc[i, 5])) * factor
    #         subj_y_relative = ((int(data.iloc[i, 6]) + int(data.iloc[i, 3])) -
    #                            (int(data.iloc[i, 13]) + int(data.iloc[i, 10]))) * factor
    #         subj_box = patches.Rectangle((0 - 0.5 * int(data.iloc[i, 4]) * factor + subj_x_relative,
    #                                      0 - 0.5 * int(data.iloc[i, 3]) * factor + subj_y_relative),
    #                                      int(data.iloc[i, 11]) * factor, int(data.iloc[i, 10]) * factor,
    #                                      linewidth=1, edgecolor='r', facecolor='none')
    #
    #         ax.add_patch(obj_box)
    #         ax.add_patch(subj_box)
    #
    #         # Keep plot graph square
    #         plt.gca().set_aspect('equal', adjustable='box')
    #         # plt.title(pred + ": " + str(i))
    #         # plt.savefig("new_pred_bb_plot/" + pred + "_bounding_box_plot_" + str(i) + ".png", dpi=50)
    #
    #         # ax = fig.gca()
    #         # ax.set_xticks(np.arange(-1, 1, 0.01))
    #         # ax.set_yticks(np.arange(-1, 1., 0.01))
    #         # plt.grid(True)
    #         # plt.rc('grid', linestyle="-", color='black')
    #         # plt.axis('off')
    #
    #         # plt.savefig("new_pred_bb_plot/" + pred + "_bounding_box_plot_" + str(i) + ".pdf", bbox_inches='tight')
    #         plt.savefig("new_pred_bb_plot/" + pred + "_bounding_box_plot_" + str(i) + ".png", dpi=55,
    #                     bbox_inches='tight')
    #         plt.show()

    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(pred + ": " + str(i))
    # plt.savefig("new_pred_plot/" + pred + "bounding_box_plot.pdf")
    # plt.show()

        # Turn plot into one long 300-dimension 1d array by __
        # Return these arrays, as well as their corresponding predicate for LDA later


def fulfills_rules(row, longest_separation, factor):
    min_pixels = 3
    if row[3] <= min_pixels or row[4] <= min_pixels or row[10] <= min_pixels or row[11] <= min_pixels:
        return False
    if longest_separation > 1:
        return False
    # if (max(row[3] * row[4], row[10] * row[11]) / min(row[3] * row[4], row[10] * row[11])) > 3:
    #     return False
    # if (longest_separation / math.sqrt(max(row[3] * row[4] * (factor ** 2), row[10] * row[11] * (factor ** 2)))) > 5:
    #     return False
    return True


def get_feature_vectors(pred):
    data = get_pred_data(pred)

    feature_vectors = []

    for i in range(len(data)):
        obj_area = (float(data.iloc[i, 5]) + float(data.iloc[i, 4])) * (float(data.iloc[i, 6]) + float(data.iloc[i, 3]))

        factor = 1 / math.sqrt(data.iloc[i, 4] * data.iloc[i, 3])
        # factor = 1

        x_diff = (float(data.iloc[i, 12]) - float(data.iloc[i, 5])) * factor
        y_diff = (float(data.iloc[i, 13]) - float(data.iloc[i, 6])) * factor
        width_diff = (float(data.iloc[i, 11]) - float(data.iloc[i, 4])) * factor
        height_diff = (float(data.iloc[i, 10]) - float(data.iloc[i, 3])) * factor
        x_cent_diff = (x_diff + (0.5 * width_diff)) * factor
        y_cent_diff = (y_diff + (0.5 * height_diff)) * factor
        area_ratio = (float(data.iloc[i, 11]) * float(data.iloc[i, 10])) / \
                     (float(data.iloc[i, 4]) * float(data.iloc[i, 3]))

        overlap_width = 0
        overlap_height = 0
        if (float(data.iloc[i, 13]) < float(data.iloc[i, 6])) and \
                (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > float(data.iloc[i, 6]):
            if (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > (float(data.iloc[i, 6]) + float(data.iloc[i, 3])):
                overlap_height = (float(data.iloc[i, 6]) - float(data.iloc[i, 13]))
            else:
                overlap_height = float(data.iloc[i, 13]) + float(data.iloc[i, 10]) - \
                            (float(data.iloc[i, 6]) - float(data.iloc[i, 13]))
        elif (float(data.iloc[i, 13]) > float(data.iloc[i, 6])) and \
                (float(data.iloc[i, 6]) + float(data.iloc[i, 3])) > float(data.iloc[i, 13]):
            if (float(data.iloc[i, 6]) + float(data.iloc[i, 3])) > (float(data.iloc[i, 13]) + float(data.iloc[i, 10])):
                overlap_height = float(data.iloc[i, 13]) + float(data.iloc[i, 10])
            else:
                overlap_height = float(data.iloc[i, 6]) + float(data.iloc[i, 3]) - \
                            (float(data.iloc[i, 13]) - float(data.iloc[i, 6]))

        if (float(data.iloc[i, 12]) < float(data.iloc[i, 5])) and \
                (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > float(data.iloc[i, 5]):
            if (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
                overlap_width = float(data.iloc[i, 5]) + float(data.iloc[i, 4])
            else:
                overlap_width = float(data.iloc[i, 12]) + float(data.iloc[i, 11]) - \
                            (float(data.iloc[i, 5]) - float(data.iloc[i, 12]))
        elif (float(data.iloc[i, 12]) > float(data.iloc[i, 5])) and \
                (float(data.iloc[i, 5]) + float(data.iloc[i, 4])) > float(data.iloc[i, 12]):
            if (float(data.iloc[i, 5]) + float(data.iloc[i, 4])) > (float(data.iloc[i, 12]) + float(data.iloc[i, 11])):
                overlap_width = float(data.iloc[i, 12]) + float(data.iloc[i, 11])
            else:
                overlap_width = float(data.iloc[i, 5]) + float(data.iloc[i, 4]) - \
                            (float(data.iloc[i, 12]) - float(data.iloc[i, 5]))

        overlap_area = overlap_width * overlap_height
        # print('overlap_area', overlap_area, 'obj_area', obj_area)
        amount_overlap = abs(overlap_area / obj_area)

        # print('amount_overlap', amount_overlap)
        if amount_overlap > 1:
            # print('something is wrong')
            return
        feature_vector = [x_diff, y_diff, x_cent_diff, y_cent_diff, amount_overlap, width_diff, height_diff, area_ratio]

        # print(feature_vector)

        feature_vectors.append(feature_vector)

    return feature_vectors


def dim_reduc_feat_vecs(predicates):
    pred_feat_vec_location = "pred_occurrences/" + predicates[0] + "_feat_vecs.p"
    feat_vecs = np.asarray(pickle.load(open(pred_feat_vec_location, "rb")))

    feat_vecs_preds = []
    for i in range(len(feat_vecs)):
        feat_vecs_preds.append(0)

    n_dimensions = 2

    # print(len(predicates))

    for i in range(1, len(predicates)):
        pred_feat_vec_location = "pred_occurrences/" + predicates[i] + "_feat_vecs.p"
        pred_feat_vecs = np.asarray(pickle.load(open(pred_feat_vec_location, "rb")))
        feat_vecs = np.concatenate((feat_vecs, pred_feat_vecs), axis=0)
        for j in range(len(pred_feat_vecs)):
            feat_vecs_preds.append(i)

        print(predicates[i] + " done")

    feat_vecs_preds = np.asarray(feat_vecs_preds)

    clf = LinearDiscriminantAnalysis(n_components=n_dimensions)
    clf.fit(feat_vecs, feat_vecs_preds)
    new_vectors = clf.transform(feat_vecs)

    print(new_vectors[0])
    print(len(new_vectors))
    print(len(feat_vecs_preds))

    return new_vectors, feat_vecs_preds


def plot_feature_vectors(vectors_file_name, vector_preds_file_name, predicates):
    # Retrieve dimensionally reduced vectors for each subj/obj/subj-obj
    label_location = "pred_occurrences/" + vectors_file_name + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve corresponding most dominant predicate for each subj/obj/subj-obj
    label_location = "pred_occurrences/" + vector_preds_file_name + ".p"
    vector_preds = np.asarray(pickle.load(open(label_location, "rb")))

    # # check when IN starts and ends
    # for i in range(len(vector_preds)):
    #     if vector_preds[i] == 26:
    #         print(i)
    #         return i

    print(vectors[0])

    # IN 114974:340327
    # ON 373986:776315
    plt.subplot(1, 2, 1)
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.plot(vectors[114974:340327, 0], vectors[114974:340327, 1], 'bo', markersize=1)
    plt.plot(vectors[373986:776315, 0], vectors[373986:776315, 1], 'ro', markersize=1)


    # Add x, y axis
    plt.axhline(linewidth=0.5, color='k')
    plt.axvline(linewidth=0.5, color='k')

    # Kernel Density Estimator
    plt.subplot(1, 2, 2)
    plt.axis([-10, 10, -10, 10])

    # kde_x_normalized_diff = vectors[114974:340327, 0]
    # kde_y_normalized_diff = vectors[114974:340327, 1]

    kde_x_normalized_diff = vectors[373986:776315, 0]
    kde_y_normalized_diff = vectors[373986:776315, 1]

    k = kde.gaussian_kde([kde_x_normalized_diff, kde_y_normalized_diff])
    xi, yi = np.mgrid[kde_x_normalized_diff.min():kde_x_normalized_diff.max():100j,
             kde_y_normalized_diff.min():kde_y_normalized_diff.max():100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_earth_r)
    plt.axhline(linewidth=0.5, color='k')
    plt.axvline(linewidth=0.5, color='k')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.colorbar()
    ###########################

    plt.show()


    # fit = stats.norm.pdf(vectors, np.mean(vectors), np.std(vectors))  # this is a fitting indeed
    #
    # pl.plot(vectors, fit, 'o')
    # # plt.title()
    # #
    # pl.hist(vectors, normed=True)  # use this to draw histogram of your data
    #
    # pl.show()



def main():
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # predicates = ['in', 'on']

    # # Plot each predicate by itself
    # for pred in predicates:
    #     outlier_img_ids = plot_normalized_centre_points(pred) #return outlier_img_ids for when want to check for weird images

    # Plot all predicates together, each with different colour
    # plot_all(predicates)

    # # Check for weird images
    # print(pred + " has " + str(len(outlier_img_ids)) + "/" + str(len(data)) + " (" + str(617/13844*100.0) + "%)"
    #       + " weird images")
    # box_images(outlier_img_ids)

    # # Box all images in order of predicates
    # box_images(predicates)

    # # Dimensionality Reduction
    # #
    # # Plot 9 plots. 3 for subj, 3 for obj, 3 for subj - obj. The 3 plots for each subj/obj/subj - obj respectively is
    # # using a different dimensionality reduction method - PCA, t-SNE, LDA respectively
    # subj_most_dominant_pred = get_most_dominant_pred('prep_subj')
    # print(len(subj_most_dominant_pred))
    # obj_most_dominant_pred = get_most_dominant_pred('prep_obj')
    # print(len(obj_most_dominant_pred))
    # subj_obj_most_dominant_pred = get_most_dominant_pred('prep_subj_obj')
    # print(len(subj_obj_most_dominant_pred))
    #
    #
    # # Save obj, subj, subj_obj, pred_order as pickle files
    # pickle_file(subj_most_dominant_pred, "subj_most_dominant_pred")
    # print("subj_most_dominant_pred saved")
    # pickle_file(obj_most_dominant_pred, "obj_most_dominant_pred")
    # print("obj_most_dominant_pred saved")
    # pickle_file(subj_obj_most_dominant_pred, "subj_obj_most_dominant_pred")
    # print("subj_obj_most_dominant_pred saved")
    #
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print("word2vec model loaded")
    # word_vectors = model.wv
    # vocab = word_vectors.vocab
    #
    # subj_vec, subj_vec_preds = word_to_vector("subj", "subj_most_dominant_pred", model, vocab)
    # print('len(subj_vec)', len(subj_vec))
    # print('len(subj_vec_preds)', len(subj_vec_preds))
    # print("subj_vec done")
    # obj_vec, obj_vec_preds = word_to_vector("obj", "obj_most_dominant_pred", model, vocab)
    # print('len(obj_vec)', len(obj_vec))
    # print('len(obj_vec_preds)', len(obj_vec_preds))
    # print("obj_vec done")
    # subj_obj_vec, subj_obj_vec_preds = word_to_vector("subj_obj", "subj_obj_most_dominant_pred", model, vocab)
    # print('len(subj_obj_vec)', len(subj_obj_vec))
    # print('len(subj_obj_vec_preds)', len(subj_obj_vec_preds))
    # print("subj_obj_vec done")
    #
    # pickle_file(subj_vec, "subj_vec")
    # print("subj_vec saved")
    # pickle_file(obj_vec, "obj_vec")
    # print("obj_vec saved")
    # pickle_file(subj_obj_vec, "subj_obj_vec")
    # print("subj_obj_vec saved")
    #
    # pickle_file(subj_vec_preds, "subj_vec_preds")
    # print("subj_vec_preds saved")
    # pickle_file(obj_vec_preds, "obj_vec_preds")
    # print("obj_vec_preds saved")
    # pickle_file(subj_obj_vec_preds, "subj_obj_vec_preds")
    # print("subj_obj_vec_preds saved")
    #
    # Reduce dimensionality for each subj/obj/subj-obj and each pca/tsne/lda
    #
    # subj_vec_reduced_pca = reduce_dimensionality('subj_vec', None, 'pca')
    # print("subj_vec_reduced_pca reduced")
    # pickle_file(subj_vec_reduced_pca, "subj_vec_reduced_pca")
    # print("subj_vec_reduced_pca saved")

    # subj_vec_reduced_tsne = reduce_dimensionality('subj_vec', None, 'tsne')
    # print("subj_vec_reduced_tsne reduced")
    # pickle_file(subj_vec_reduced_tsne, "subj_vec_reduced_tsne")
    # print("subj_vec_reduced_tsne saved")
    #
    # subj_vec_reduced_lda = reduce_dimensionality('subj_vec', 'subj_vec_preds', 'lda')
    # print("subj_vec_reduced_lda reduced")
    # pickle_file(subj_vec_reduced_lda, "subj_vec_reduced_lda")
    # print("subj_vec_reduced_lda saved")
    #
    # obj_vec_reduced_pca = reduce_dimensionality('obj_vec', None, 'pca')
    # print("obj_vec_reduced_pca reduced")
    # pickle_file(obj_vec_reduced_pca, "obj_vec_reduced_pca")
    # print("obj_vec_reduced_pca saved")

    # obj_vec_reduced_tsne = reduce_dimensionality('obj_vec', None, 'tsne')
    # print("obj_vec_reduced_tsne reduced")
    # pickle_file(obj_vec_reduced_tsne, "obj_vec_reduced_tsne")
    # print("obj_vec_reduced_tsne saved")
    #
    # obj_vec_reduced_lda = reduce_dimensionality('obj_vec', 'obj_vec_preds', 'lda')
    # print("obj_vec_reduced_lda reduced")
    # pickle_file(obj_vec_reduced_lda, "obj_vec_reduced_lda")
    # print("obj_vec_reduced_lda saved")
    #
    # subj_obj_vec_reduced_pca = reduce_dimensionality('subj_obj_vec', None, 'pca')
    # print("sbj_obj_vec_reduced_pca reduced")
    # pickle_file(subj_obj_vec_reduced_pca, "subj_obj_vec_reduced_pca")
    # print("subj_obj_vec_reduced_pca saved")

    # subj_obj_vec_reduced_tsne = reduce_dimensionality('subj_obj_vec', None, 'tsne')
    # print("subj_obj_vec_reduced_tsne reduced")
    # pickle_file(subj_obj_vec_reduced_tsne, "subj_obj_vec_reduced_tsne")
    # print("subj_obj_vec_reduced_tsne saved")
    #
    # subj_obj_vec_reduced_lda = reduce_dimensionality('subj_obj_vec', 'subj_obj_vec_preds', 'lda')
    # print("subj_obj_vec_reduced_lda reduced")
    # pickle_file(subj_obj_vec_reduced_lda, "subj_obj_vec_reduced_lda")
    # print("subj_obj_vec_reduced_lda saved")

    # Plot all dimensionality reduced plots

    # plot_vectors("subj_vec_reduced_pca", "subj_vec_preds", "Subjects: PCA", "2", predicates)
    # plot_vectors("obj_vec_reduced_pca", "obj_vec_preds", "Objects: PCA", "2", predicates)
    # plot_vectors("subj_obj_vec_reduced_pca", "subj_obj_vec_preds", "Subjects - Objects: PCA", "2", predicates)
    #
    # plot_vectors("subj_vec_reduced_tsne", "subj_vec_preds", "Subjects: t-SNE", "2", predicates)
    # plot_vectors("obj_vec_reduced_tsne", "obj_vec_preds", "Objects: t-SNE", "2", predicates)
    # plot_vectors("subj_obj_vec_reduced_tsne", "subj_obj_vec_preds", "Subjects - Objects: t-SNE", "2", predicates)
    #
    # plot_vectors("subj_vec_reduced_lda", "subj_vec_preds", "Subjects: LDA", "2", predicates)
    # plot_vectors("obj_vec_reduced_lda", "obj_vec_preds", "Objects: LDA", "2", predicates)
    # plot_vectors("subj_obj_vec_reduced_fda", "subj_obj_vec_preds", "Subjects - Objects: LDA", "2", predicates)

    # plot_vectors("subj_vec_reduced_pca", "subj_vec_preds", "Subjects: PCA", "3", predicates)
    # plot_vectors("obj_vec_reduced_pca", "obj_vec_preds", "Objects: PCA", "3", predicates)
    # plot_vectors("subj_obj_vec_reduced_pca", "subj_obj_vec_preds", "Subjects - Objects: PCA", "3", predicates)
    #
    # plot_vectors("subj_vec_reduced_tsne", "subj_vec_preds", "Subjects: t-SNE", "3", predicates)
    # plot_vectors("obj_vec_reduced_tsne", "obj_vec_preds", "Objects: t-SNE", "3", predicates)
    # plot_vectors("subj_obj_vec_reduced_tsne", "subj_obj_vec_preds", "Subjects - Objects: t-SNE", "3", predicates)
    #
    # plot_vectors("subj_vec_reduced_lda", "subj_vec_preds", "Subjects: LDA", "3", predicates)
    # plot_vectors("obj_vec_reduced_lda", "obj_vec_preds", "Objects: LDA", "3", predicates)
    # plot_vectors("subj_obj_vec_reduced_lda", "subj_obj_vec_preds", "Subjects - Objects: LDA", "3", predicates)
    #
    #
    # # Bounding Box Method
    #
    # Get longest edge of all subjs and objs of all predicates
    # longest_separation, longest_separation_pred, longest_separation_pred_index, longest_separation_img_id \
    #     = find_longest_separation(predicates)
    # longest_edge = 1281
    # longest_separation = 1126.0
    # print('find_longest(predicates) done')
    # print('longest_separation', longest_separation)
    # print('longest_separation_pred', longest_separation_pred)
    # print('longest_separation_pred_index', longest_separation_pred_index)
    # print('longest_separation_img_id', longest_separation_img_id)

    # decrease_factor = 0.5 / longest_edge
    # normalized_longest_separation = longest_separation * decrease_factor
    # print('decrease_factor', decrease_factor)
    # print('normalized_longest_separation', normalized_longest_separation)

    # for pred in predicates:
    #     get_bounding_box_vectors(pred, decrease_factor, normalized_longest_separation)
    # print('longest_edge', longest_edge)

    # for pred in predicates:
    #     get_normalized_bounding_box_vectors(pred, 1)
    # print('longest_edge', longest_edge)


    # For presentation

    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print("word2vec model loaded")
    # # word_vectors = model.wv
    # # vocab = word_vectors.vocab
    #
    # print("cup", model["cup"])
    # print(len(model["cup"]))
    # print("table", model["table"])
    # print(len(model["table"]))
    # # subj_vec, subj_vec_preds = word_to_vector("subj", "subj_most_dominant_pred", model, vocab)

    # # Plot individual normalized points ##################
    #
    # pred = "above"
    # data = get_pred_data(pred)
    # i = 53
    # obj_centre_x = float(data.iloc[i, 5]) + (float(data.iloc[i, 4]) / 2.0)
    # obj_centre_y = float(data.iloc[i, 6]) + (float(data.iloc[i, 3]) / 2.0)
    # subj_centre_x = float(data.iloc[i, 12]) + (float(data.iloc[i, 11]) / 2.0)
    # subj_centre_y = float(data.iloc[i, 13]) + (float(data.iloc[i, 10]) / 2.0)
    #
    # x_normalized_diff = (subj_centre_x - obj_centre_x) / float(data.iloc[i, 15])
    # y_normalized_diff = (obj_centre_y - subj_centre_y) / float(data.iloc[i, 16])
    #
    # plt.figure(figsize=(5, 5))
    # plt.title(pred)
    # plt.plot(x_normalized_diff, y_normalized_diff, 'ro', markersize=3)
    # plt.plot(0, 0, 'bo', markersize=3)
    # plt.axis([-1, 1, -1, 1])
    #
    # # Add x, y axis
    # plt.axhline(linewidth=0.5, color='k')
    # plt.axvline(linewidth=0.5, color='k')
    #
    # # Keep plot graph square
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("../Presentation-Images/" + pred + "_" + str(i) + "_centroid.png", dpi=300)
    #
    # plt.show()
    #
    # ##############################################

    # for pred in predicates:
    #     feature_vectors = get_feature_vectors(pred)
    #     pickle_file(feature_vectors, pred + "_feat_vecs")
    #     print(pred + " feature vectors saved")
    #
    # dim_reduce_feature_vectors, dim_reduce_feature_vectors_preds = dim_reduc_feat_vecs(predicates)
    # pickle_file(dim_reduce_feature_vectors, "dim_reduce_feature_vectors")
    # pickle_file(dim_reduce_feature_vectors_preds, "dim_reduce_feature_vectors_preds")
    #
    # plot_vectors("dim_reduce_feature_vectors", "dim_reduce_feature_vectors_preds", 'hi', '2', predicates)
    plot_feature_vectors("dim_reduce_feature_vectors", "dim_reduce_feature_vectors_preds", predicates)
    # print('hi')


if __name__ == "__main__":
    main()
