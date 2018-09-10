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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import math
import scipy.stats as stats
import pylab as pl
# import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import random


def get_prep_data(prep):
    file_address = 'new_prep/data_prep_' + prep + '.csv'
    data = pd.read_csv(file_address, encoding="ISO-8859-1")
    return data


def get_obj_and_subj(prepositions):
    subj = []
    obj = []
    subj_obj = []

    # Loop through data rows and add new subjects and objects to subjects and objects unique list
    for prep in prepositions:
        data = get_prep_data(prep)
        for i in range(len(data)):
            if data.iloc[i, 2] not in obj:
                # print("data.iloc[i, 2]:", data.iloc[i, 2])
                obj.append(data.iloc[i, 2])
            if data.iloc[i, 9] not in subj:
                # print("data.iloc[i, 9]:", data.iloc[i, 9])
                subj.append(data.iloc[i, 9])
            if (data.iloc[i, 9], data.iloc[i, 2]) not in subj_obj:
                # print("data.iloc[i, 9], data.iloc[i, 2]:", data.iloc[i,9], ",", data.iloc[i,2])
                subj_obj.append((data.iloc[i, 9], data.iloc[i, 2]))
        print("get_obj_and_subj:", prep, "done")

    return subj, obj, subj_obj


def fill_tables(obj_file, subj_file, subj_obj_file, prepositions):
    # Fill tables of obj, subj, and subj_obj where each row has counts of the number of appearances of each preposition
    # for each unique thing

    subj_file_location = "prep_occurrences/" + subj_file + ".p"
    subj = pickle.load(open(subj_file_location, "rb"))

    obj_file_location = "prep_occurrences/" + obj_file + ".p"
    obj = pickle.load(open(obj_file_location, "rb"))

    subj_obj_file_location = "prep_occurrences/" + subj_obj_file + ".p"
    subj_obj = pickle.load(open(subj_obj_file_location, "rb"))

    prep_subj = np.zeros((len(subj), len(prepositions)))
    prep_obj = np.zeros((len(obj), len(prepositions)))
    prep_subj_obj = np.zeros((len(subj_obj), len(prepositions)))

    for prep in prepositions:
        data = get_prep_data(prep)
        for i in range(len(data)):
            prep_subj[subj.index(data.iloc[i, 9]), prepositions.index(prep)] += 1
            prep_obj[obj.index(data.iloc[i, 2]), prepositions.index(prep)] += 1
            prep_subj_obj[subj_obj.index((data.iloc[i, 9], data.iloc[i, 2])), prepositions.index(prep)] += 1
        print("fill_tables:", prep, "done")

    return prep_subj, prep_obj, prep_subj_obj


def get_most_dominant_prep(table_name):
    # Retrieve counts for each prep occurrence for each item in table
    table_location = "prep_occurrences/" + table_name + ".p"
    table = pickle.load(open(table_location, "rb"))

    most_dominant_prep = []

    # Loop through data rows and add index of largest integer for each row (most dominant preposition for this row)
    for i in range(len(table)):
        most_dominant_prep.append(np.argmax(table[i]))

    return most_dominant_prep


def plot_normalized_centre_points(prep):
    # Get obj and subj centre points and normalize diffs
    x_normalized_diff = []
    y_normalized_diff = []

    # Weird images
    outlier_relationships = []

    data = get_prep_data(prep)

    for i in range(len(data)):
        obj_centre_x = float(data.iloc[i, 5]) + (float(data.iloc[i, 4]) / 2.0)
        obj_centre_y = float(data.iloc[i, 6]) + (float(data.iloc[i, 3]) / 2.0)
        subj_centre_x = float(data.iloc[i, 12]) + (float(data.iloc[i, 11]) / 2.0)
        subj_centre_y = float(data.iloc[i, 13]) + (float(data.iloc[i, 10]) / 2.0)

        x_normalized_diff.append((subj_centre_x - obj_centre_x) / float(data.iloc[i, 15]))
        y_normalized_diff.append((obj_centre_y - subj_centre_y) / float(data.iloc[i, 16]))

        # For specific prepositions #
        if prep == 'above':
            if y_normalized_diff[-1] < 0:
                outlier_relationships.append(data.iloc[i, ])
        elif prep == 'below':
            if y_normalized_diff[-1] > 0:
                outlier_relationships.append(data.iloc[i, ])
        ###########################

    # Graph the normalized diff coordinates
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.title(prep)
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

    plt.title(prep)
    plt.colorbar()

    # Save plots
    plt.savefig("new_prep_plot/" + prep + "_plot.png", dpi=300)

    # plt.show()

    return outlier_relationships


def plot_all(prepositions):

    # plt.figure(figsize=(14, 5))
    # plt.subplot(1, 2, 1)
    plt.title("All")
    plt.axis([-1, 1, -1, 1])
    # Add x, y axis
    plt.axhline(linewidth=0.5, color='k')
    plt.axvline(linewidth=0.5, color='k')

    # Keep plot graph square
    plt.gca().set_aspect('equal', adjustable='box')
    for prep in prepositions:
        # Get obj and subj centre points and normalize diffs
        x_normalized_diff = []
        y_normalized_diff = []

        data = get_prep_data(prep)

        for i in range(len(data)):
            obj_centre_x = float(data.iloc[i, 5]) + (float(data.iloc[i, 4]) / 2.0)
            obj_centre_y = float(data.iloc[i, 6]) + (float(data.iloc[i, 3]) / 2.0)
            subj_centre_x = float(data.iloc[i, 12]) + (float(data.iloc[i, 11]) / 2.0)
            subj_centre_y = float(data.iloc[i, 13]) + (float(data.iloc[i, 10]) / 2.0)

            x_normalized_diff.append((subj_centre_x - obj_centre_x) / float(data.iloc[i, 15]))
            y_normalized_diff.append((obj_centre_y - subj_centre_y) / float(data.iloc[i, 16]))

        # Graph the normalized diff coordinates
        plt.plot(x_normalized_diff, y_normalized_diff, 'o', markersize=1)

        print(prep, "done")

    # # Save plot
    # plt.savefig("new_prep_plot/combined_one_plot.pdf", dpi=300)

    plt.show()


def box_images(prep, i):
    # Box specific images

    data = get_prep_data(prep)

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

    # # Add centroids
    # plt.plot(int(data.iloc[i, 5]) + (0.5 * int(data.iloc[i, 4])),
    #          int(data.iloc[i, 6]) + (0.5 * int(data.iloc[i, 3])), 'bo', markersize=3)
    # plt.plot(int(data.iloc[i, 12]) + (0.5 * int(data.iloc[i, 11])),
    #          int(data.iloc[i, 13]) + (0.5 * int(data.iloc[i, 10])), 'ro', markersize=3)

    # Label obj and subj
    plt.annotate("Object: " + data.iloc[i, 2], (int(data.iloc[i, 5]), int(data.iloc[i, 6])), color='k')
    plt.annotate("Subject: " + data.iloc[i, 9], (int(data.iloc[i, 12]), int(data.iloc[i, 13])), color='k')

    plt.title(prep + ": " + str(i))
    plt.show()

    # for i in range(len(data)):
    #     # print(max(data.iloc[i, 3] * data.iloc[i, 4], data.iloc[i, 10] * data.iloc[i, 11]))
    #     image_address = 'images/' + str(data.iloc[i, 0]) + '.jpg'
    #     im = np.array(Image.open(image_address), dtype=np.uint8)
    #
    #     # Create figure and axes
    #     fig, ax = plt.subplots(1)
    #
    #     # Display the image
    #     ax.imshow(im)
    #
    #     # Create a Rectangle patch
    #     obj_box = patches.Rectangle((int(data.iloc[i, 5]), int(data.iloc[i, 6])), int(data.iloc[i, 4]),
    #                                 int(data.iloc[i, 3]), linewidth=1, edgecolor='b', facecolor='none')
    #     subj_box = patches.Rectangle((int(data.iloc[i, 12]), int(data.iloc[i, 13])), int(data.iloc[i, 11]),
    #                                  int(data.iloc[i, 10]), linewidth=1, edgecolor='r', facecolor='none')
    #
    #     # Add the patch to the Axes
    #     ax.add_patch(obj_box)
    #     ax.add_patch(subj_box)
    #
    #     # # Add centroids
    #     # plt.plot(int(data.iloc[i, 5]) + (0.5 * int(data.iloc[i, 4])),
    #     #          int(data.iloc[i, 6]) + (0.5 * int(data.iloc[i, 3])), 'bo', markersize=3)
    #     # plt.plot(int(data.iloc[i, 12]) + (0.5 * int(data.iloc[i, 11])),
    #     #          int(data.iloc[i, 13]) + (0.5 * int(data.iloc[i, 10])), 'ro', markersize=3)
    #
    #
    #     # Label obj and subj
    #     plt.annotate("Object: " + data.iloc[i, 2], (int(data.iloc[i, 5]), int(data.iloc[i, 6])), color='k')
    #     plt.annotate("Subject: " + data.iloc[i, 9], (int(data.iloc[i, 12]), int(data.iloc[i, 13])), color='k')
    #
    #
    #     # Colour in overlap area
    #     subj_area = float(data.iloc[i, 11]) * float(data.iloc[i, 10])
    #
    #     overlap_width = 0
    #     overlap_height = 0
    #     overlap_x = 0
    #     overlap_y = 0
    #     # subj_y <= obj_y and subj_y + subj_h > obj_y
    #     if (float(data.iloc[i, 13]) <= float(data.iloc[i, 6])) and \
    #             (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > float(data.iloc[i, 6]):
    #         overlap_y = data.iloc[i, 6]
    #         # subj_y + subj_h > obj_y + obj_h
    #         if (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > (float(data.iloc[i, 6]) + float(data.iloc[i, 3])):
    #             overlap_height = float(data.iloc[i, 3])
    #         else:
    #             overlap_height = float(data.iloc[i, 13]) + float(data.iloc[i, 10]) - float(data.iloc[i, 6])
    #     # subj_y > obj_y and subj_y < obj_y + obj_h
    #     elif (float(data.iloc[i, 13]) > float(data.iloc[i, 6])) and \
    #             float(data.iloc[i, 13]) < float(data.iloc[i, 6]) + float(data.iloc[i, 3]):
    #         overlap_y = data.iloc[i, 13]
    #         # subj_y + subj_h > obj_y + obj_h
    #         if (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > (float(data.iloc[i, 6]) + float(data.iloc[i, 3])):
    #             overlap_height = float(data.iloc[i, 6]) + float(data.iloc[i, 3]) - float(data.iloc[i, 13])
    #         else:
    #             overlap_height = float(data.iloc[i, 10])
    #
    #     # subj_x <= obj_x and subj_x + subj_w > obj_x
    #     if (float(data.iloc[i, 12]) <= float(data.iloc[i, 5])) and \
    #             (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > float(data.iloc[i, 5]):
    #         overlap_x = data.iloc[i, 5]
    #         # subj_x + subj_w > obj_x + obj_w
    #         if (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
    #             overlap_width = float(data.iloc[i, 4])
    #         else:
    #             overlap_width = float(data.iloc[i, 12]) + float(data.iloc[i, 11]) - float(data.iloc[i, 5])
    #     # subj_x > obj_x and subj_x < obj_x + obj_w
    #     elif (float(data.iloc[i, 12]) > float(data.iloc[i, 5])) and \
    #             float(data.iloc[i, 12]) < (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
    #         overlap_x = data.iloc[i, 12]
    #         # subj_x + subj_w > obj_x + obj_w
    #         if (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
    #             overlap_width = (float(data.iloc[i, 5]) + float(data.iloc[i, 4])) - float(data.iloc[i, 12])
    #         else:
    #             overlap_width = float(data.iloc[i, 11])
    #
    #     overlap_area = overlap_width * overlap_height
    #     amount_overlap = abs(overlap_area / subj_area)
    #
    #     rect1 = patches.Rectangle((overlap_x, overlap_y), overlap_width, overlap_height, color='yellow')
    #     # ax.add_patch(rect1)
    #     print(str(i) + ": " + str(amount_overlap))
    #     plt.title(prep + ' overlap: ' + str(amount_overlap))
    #     plt.show()

    # # Box all prepositions
    # for prep in prepositions:
    #     data = get_prep_data(prep)
    #     for i in range(0, len(data)):
    #         # Decide whether row i fulfills rules #
    #         factor = math.sqrt(0.5 / (int(data.iloc[i, 3] * int(data.iloc[i, 4]))))
    #
    #         # get max y
    #         y_diff_1 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - data.iloc[i, 13])
    #         y_diff_2 = abs(data.iloc[i, 6] + (0.5 * data.iloc[i, 3]) - (data.iloc[i, 13] + data.iloc[i, 10]))
    #         max_y = max(y_diff_1, y_diff_2, 0.5 * data.iloc[i, 3])
    #
    #         # get max x
    #         x_diff_1 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - data.iloc[i, 12])
    #         x_diff_2 = abs(data.iloc[i, 5] + (0.5 * data.iloc[i, 4]) - (data.iloc[i, 12] + data.iloc[i, 11]))
    #         max_x = max(x_diff_1, x_diff_2, 0.5 * data.iloc[i, 4])
    #
    #         # max separation
    #         longest_separation = max(max_y * factor, max_x * factor)
    #
    #         #######################################
    #         if fulfills_rules(data.iloc[i,], longest_separation, factor):
    #             image_address = 'images/' + str(data.iloc[i, 0]) + '.jpg'
    #             im = np.array(Image.open(image_address), dtype=np.uint8)
    #
    #             # Create figure and axes
    #             fig, ax = plt.subplots(1)
    #
    #             # Display the image
    #             ax.imshow(im)
    #
    #             # Create a Rectangle patch
    #             obj_box = patches.Rectangle((int(data.iloc[i, 5]), int(data.iloc[i, 6])), int(data.iloc[i, 4]),
    #                                         int(data.iloc[i, 3]), linewidth=1, edgecolor='b', facecolor='none')
    #             subj_box = patches.Rectangle((int(data.iloc[i, 12]), int(data.iloc[i, 13])), int(data.iloc[i, 11]),
    #                                          int(data.iloc[i, 10]), linewidth=1, edgecolor='r', facecolor='none')
    #
    #             # Add the patch to the Axes
    #             ax.add_patch(obj_box)
    #             ax.add_patch(subj_box)
    #
    #             # Label obj and subj
    #             plt.annotate("Object: " + data.iloc[i, 2], (int(data.iloc[i, 5]), int(data.iloc[i, 6])), color='k')
    #             plt.annotate("Subject: " + data.iloc[i, 9], (int(data.iloc[i, 12]), int(data.iloc[i, 13])), color='w')
    #
    #             plt.title("Preposition: " + prep)
    #             # plt.title(prep + ": " + str(i))
    #             print(str(i))
    #             plt.show()


def word_to_vector(table_name, prep_order_name, model, vocab):
    # Retrieve labels for each row in table
    label_location = "prep_occurrences/" + table_name + ".p"
    words = pickle.load(open(label_location, "rb"))

    prep_order_location = "prep_occurrences/" + prep_order_name + ".p"
    preps = pickle.load(open(prep_order_location, "rb"))

    # Only keep words that are part of pre-trained word2vec vocabulary
    word_vectors = []
    new_prep_order = []
    for i in range(len(words)):
        # if (subj, obj)
        if isinstance(words[i], tuple):
            if words[i][0] in vocab and words[i][1] in vocab:
                # print(np.asarray(model[words[i][0]]))
                concatenated_word_vectors = np.concatenate((np.asarray(model[words[i][0]]), np.asarray(model[words[i][1]])))
                # print(len(concatenated_word_vectors))
                # print(np.concatenate(np.asarray(model[words[i][0]]), np.asarray(model[words[i][1]])))
                word_vectors.append(concatenated_word_vectors)
                new_prep_order.append(preps[i])
        # if subj, or obj
        else:
            if words[i] in vocab:
                word_vectors.append(model[words[i]])
                new_prep_order.append(preps[i])

    return word_vectors, new_prep_order


def pickle_file(table, name):
    # https://wiki.python.org/moin/UsingPickle
    file_name = "prep_occurrences/" + name + ".p"
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(table, fp)


def reduce_dimensionality(vector_names, dominant_prep, reduction_method):
    # Retrieve vectors for each word
    label_location = "prep_occurrences/" + vector_names + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve prepositions associated with each word only if reduction_method is 'lda'
    vector_prepositions = None
    if dominant_prep is not None:
        prep_order_location = "prep_occurrences/" + dominant_prep + ".p"
        vector_prepositions = np.asarray(pickle.load(open(prep_order_location, "rb")))

    # print('len(vectors)', len(vectors))
    # print('len(vector_prepositions)', len(vector_prepositions))

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
        clf.fit(vectors, vector_prepositions)
        new_vectors = clf.transform(vectors)
        print('lda new_vectors', new_vectors.shape)

    return new_vectors


def plot_vectors(vectors_file_name, vector_preps_file_name, title, n_dimensions, prepositions):
    # Retrieve dimensionally reduced vectors for each subj/obj/subj-obj
    label_location = "prep_occurrences/" + vectors_file_name + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve corresponding most dominant preposition for each subj/obj/subj-obj
    label_location = "prep_occurrences/" + vector_preps_file_name + ".p"
    vector_preps = np.asarray(pickle.load(open(label_location, "rb")))

    print('hi1')
    print(len(vectors))
    print(len(vector_preps))
    print(vectors[0, 0], vectors[0, 1])
    print(vectors[1, 0], vectors[1, 1])
    print(vectors[2, 0], vectors[2, 1])
    print(vectors)
    # plt.plot(vectors[0, 0], vectors[0, 1], 'bo', markersize=10)
    # plt.show()

    # Colors for each preposition
    # prep_to_color = {1: [prepositions[1], 'black'], 11: [prepositions[11], 'green'], 17: [prepositions[17], 'violet'],
    #                  20: [prepositions[20], 'blue'], 23: [prepositions[23], 'orange'], 25: [prepositions[25], 'red'],
    #                  37: [prepositions[38], 'indigo'], 41: [prepositions[41], 'yellow']}

    # print(prepositions)

    # # Colors for each preposition (reduced list)
    # prep_to_color = {0: [prepositions[0], 'blue'], 1: [prepositions[1], 'red']}
    #
    # preps = []
    # colors = []
    # for i in range(len(prepositions)):
    #     if i in prep_to_color:
    #         preps.append(prep_to_color[i][0])
    #         colors.append(prep_to_color[i][1])
    #
    # print('preps', preps)
    # print('colors', colors)
    #
    # handles_patches = [mpatches.Patch(color=colors[i], label="{:s}".format(preps[i])) for i in range(len(preps))]
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

        for i in range(len(vectors)):
            if vector_preps[i] == 20 or vector_preps[i] == 25:
                if vector_preps[i] == 20:
                    plt.plot(vectors[i, 0], vectors[i, 1], 'bo', markersize=1)
                # if vector_preps[i] == 25:
                #     plt.plot(vectors[i, 0], vectors[i, 1], 'ro', markersize=1)
                    kde_x.append(vectors[i, 0])
                    kde_y.append((vectors[i, 1]))

        # IN
        # plt.plot(vectors[114974:340327, 0], vectors[114974:340327, 1], 'bo', markersize=1)
        # kde_x.append(vectors[114974:340327, 0])
        # kde_y.append((vectors[114974:340327, 1]))

        # # ON
        # plt.plot(vectors[373986:776315, 0], vectors[373986:776315, 1], 'ro', markersize=1)
        # kde_x.append(vectors[373986:776315, 0])
        # kde_y.append((vectors[373986:776315, 1]))
        #
        # print('hi2')
        # plt.title(title + ' 2d')
        # # plt.axis([-1000, 1000, -1000, 1000])
        plt.gca().set_aspect('equal', adjustable='box')
        #
        # Add x, y axis
        plt.axhline(linewidth=0.5, color='k')
        plt.axvline(linewidth=0.5, color='k')
        #
        # # Legend
        # blue_patch = mpatches.Patch(color='blue', label='in')
        # red_patch = mpatches.Patch(color='red', label='on')
        # plt.legend(handles=[blue_patch, red_patch])
        #
        # # plt.savefig("word_vector_plot/" + vectors_file_name + "_plot_2d.pdf")

        # Kernel Density Estimator #
        plt.subplot(1, 2, 2)
        plt.axis([-2, 2, -2, 2])

        kde_x_normalized_diff = np.asarray(kde_x)
        kde_y_normalized_diff = np.asarray(kde_y)

        k = kde.gaussian_kde([kde_x_normalized_diff, kde_y_normalized_diff])
        xi, yi = np.mgrid[kde_x_normalized_diff.min():kde_x_normalized_diff.max():100j,
                 kde_y_normalized_diff.min():kde_y_normalized_diff.max():100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.gist_earth_r)
        plt.axhline(linewidth=0.5, color='k')
        plt.axvline(linewidth=0.5, color='k')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        # ############################

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
    #         if vector_preps[i] in prep_to_color:
    #             plt.plot([vectors[i, 0]], [vectors[i, 1]], [vectors[i, 2]], marker='o', markersize=1,
    #                      color=prep_to_color[vector_preps[i]][1])
    #
    #     # ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=0.1, c='b')
    #     plt.title(title + ' 3d')
    #     plt.savefig("word_vector_plot/" + vectors_file_name + "_plot_3d.pdf")


def find_longest_separation(prepositions):
    # might also want to look at which image contains longest_edge and longest_separation
    # so get image id too when u update max

    # longest_separation = 0
    # prep = 'in'
    # i = 179
    #
    # data = get_prep_data(prep)
    #
    # # get longest edge
    # old_longest_edge = longest_edge
    # longest_edge = max(longest_edge, data.iloc[i, 3], data.iloc[i, 4], data.iloc[i, 10], data.iloc[i, 11])
    # if longest_edge > old_longest_edge:
    #     longest_edge_prep = prep
    #     longest_edge_prep_index = i
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
    #     longest_separation_prep = prep
    #     longest_separation_prep_index = i
    #     longest_separation_img_id = data.iloc[i, 0]
    #
    # print(prep, 'done')
    #
    # return longest_edge, longest_separation, longest_edge_prep, longest_edge_prep_index, longest_edge_img_id, \
    #        longest_separation_prep, longest_separation_prep_index, longest_separation_img_id

    longest_separation = 0
    # longest_separation_prep = prepositions[0]
    # longest_separation_prep_index = 0
    # longest_separation_img_id = 0

    total_lines_of_data = 0
    total_used_lines_of_data = 0
    for prep in prepositions:
        data = get_prep_data(prep)
        prep_lines_of_data = len(data)
        prep_used_lines_of_data = 0
        total_lines_of_data += prep_lines_of_data
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
                    longest_separation_prep = prep
                    longest_separation_prep_index = i
                    longest_separation_img_id = data.iloc[i, 0]
                prep_used_lines_of_data += 1

        total_used_lines_of_data += prep_used_lines_of_data

        print('prep_lines_of_data', prep_lines_of_data)
        print('prep_used_lines_of_data', prep_used_lines_of_data)
        print('percent used', prep_used_lines_of_data / prep_lines_of_data)
        print(prep, 'done')

    print('total_lines_of_data', total_lines_of_data)
    print('used_lines_of_data', total_used_lines_of_data)
    print('percent used', total_used_lines_of_data / total_lines_of_data)

    return longest_separation, longest_separation_prep, longest_separation_prep_index, longest_separation_img_id


def get_bounding_box_vectors(prep, decrease_factor, plot_size):
    # # Get bounding box of specific image for specific preposition
    # prep = 'on'
    # i = 35610
    #
    # data = get_prep_data(prep)
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
    # plt.title(prep + ": " + str(data.iloc[i, 0]))
    # plt.show()

    # Get bounding boxes for all images of given preposition
    data = get_prep_data(prep)

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
        plt.title(prep + ": " + str(i))
        plt.show()

        # Turn plot into one long 300-dimension 1d array by __

        # Return these arrays, as well as their corresponding preposition for LDA later


def get_normalized_bounding_box_vectors(prep, plot_size):
    # Get bounding box of specific image for specific preposition
    prep = 'above'
    i = 53

    data = get_prep_data(prep)

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

    # plt.title(prep + ": " + str(i))
    plt.title(prep)
    plt.savefig("../Presentation-Images/" + prep + "_" + str(i)+ "_bb_plot_grid.png", dpi=2400)
    plt.show()

    # # Get bounding boxes for all images of given preposition
    # data = get_prep_data(prep)
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
    #         # plt.title(prep + ": " + str(i))
    #         # plt.savefig("new_prep_bb_plot/" + prep + "_bounding_box_plot_" + str(i) + ".png", dpi=50)
    #
    #         # ax = fig.gca()
    #         # ax.set_xticks(np.arange(-1, 1, 0.01))
    #         # ax.set_yticks(np.arange(-1, 1., 0.01))
    #         # plt.grid(True)
    #         # plt.rc('grid', linestyle="-", color='black')
    #         # plt.axis('off')
    #
    #         # plt.savefig("new_prep_bb_plot/" + prep + "_bounding_box_plot_" + str(i) + ".pdf", bbox_inches='tight')
    #         plt.savefig("new_prep_bb_plot/" + prep + "_bounding_box_plot_" + str(i) + ".png", dpi=55,
    #                     bbox_inches='tight')
    #         plt.show()

    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(prep + ": " + str(i))
    # plt.savefig("new_prep_plot/" + prep + "bounding_box_plot.pdf")
    # plt.show()

        # Turn plot into one long 300-dimension 1d array by __
        # Return these arrays, as well as their corresponding preposition for LDA later


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


def get_feature_vector(data, i):

    subj_area = float(data.iloc[i, 11]) * float(data.iloc[i, 10])

    factor = 1 / math.sqrt(data.iloc[i, 4] * data.iloc[i, 3])

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
    # subj_y < obj_y and subj_y + subj_h > obj_y
    if (float(data.iloc[i, 13]) < float(data.iloc[i, 6])) and \
            (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > float(data.iloc[i, 6]):
        # subj_y + subj_h > obj_y + obj_h
        if (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > (float(data.iloc[i, 6]) + float(data.iloc[i, 3])):
            overlap_height = float(data.iloc[i, 3])
        else:
            overlap_height = float(data.iloc[i, 13]) + float(data.iloc[i, 10]) - float(data.iloc[i, 6])
    # subj_y > obj_y and subj_y < obj_y + obj_h
    elif (float(data.iloc[i, 13]) > float(data.iloc[i, 6])) and \
            float(data.iloc[i, 13]) < float(data.iloc[i, 6]) + float(data.iloc[i, 3]):
        # subj_y + subj_h > obj_y + obj_h
        if (float(data.iloc[i, 13]) + float(data.iloc[i, 10])) > (float(data.iloc[i, 6]) + float(data.iloc[i, 3])):
            overlap_height = float(data.iloc[i, 6]) + float(data.iloc[i, 3]) - float(data.iloc[i, 13])
        else:
            overlap_height = float(data.iloc[i, 10])

    # subj_x < obj_x and subj_x + subj_w > obj_x
    if (float(data.iloc[i, 12]) < float(data.iloc[i, 5])) and \
            (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > float(data.iloc[i, 5]):
        # subj_x + subj_w > obj_x + obj_w
        if (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
            overlap_width = float(data.iloc[i, 4])
        else:
            overlap_width = float(data.iloc[i, 12]) + float(data.iloc[i, 11]) - float(data.iloc[i, 5])
    # subj_x > obj_x and subj_x < obj_x + obj_w
    elif (float(data.iloc[i, 12]) > float(data.iloc[i, 5])) and \
            float(data.iloc[i, 12]) < (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
        # subj_x + subj_w > obj_x + obj_w
        if (float(data.iloc[i, 12]) + float(data.iloc[i, 11])) > (float(data.iloc[i, 5]) + float(data.iloc[i, 4])):
            overlap_width = (float(data.iloc[i, 5]) + float(data.iloc[i, 4])) - float(data.iloc[i, 12])
        else:
            overlap_width = float(data.iloc[i, 11])

    overlap_area = overlap_width * overlap_height
    # print('overlap_area', overlap_area, 'obj_area', obj_area)
    amount_overlap = abs(overlap_area / subj_area)

    # print('amount_overlap', amount_overlap)
    if amount_overlap > 1 or amount_overlap < 0:
        print('something is wrong')
        return
    feature_vector = np.asarray([x_diff, y_diff, x_cent_diff, y_cent_diff, amount_overlap, width_diff, height_diff, area_ratio])

    return feature_vector


def dim_reduc_feat_vecs(prepositions):
    prep_feat_vec_location = "prep_occurrences/" + prepositions[0] + "_feat_vecs.p"
    feat_vecs = np.asarray(pickle.load(open(prep_feat_vec_location, "rb")))

    feat_vecs_preps = []
    for i in range(len(feat_vecs)):
        feat_vecs_preps.append(0)

    n_dimensions = 2

    # print(len(prepositions))

    for i in range(1, len(prepositions)):
        prep_feat_vec_location = "prep_occurrences/" + prepositions[i] + "_feat_vecs.p"
        prep_feat_vecs = np.asarray(pickle.load(open(prep_feat_vec_location, "rb")))
        feat_vecs = np.concatenate((feat_vecs, prep_feat_vecs), axis=0)
        for j in range(len(prep_feat_vecs)):
            feat_vecs_preps.append(i)

        print(prepositions[i] + " done")

    feat_vecs_preps = np.asarray(feat_vecs_preps)

    clf = LinearDiscriminantAnalysis(n_components=n_dimensions)
    clf.fit(feat_vecs, feat_vecs_preps)
    new_vectors = clf.transform(feat_vecs)

    print(new_vectors[0])
    print(len(new_vectors))
    print(len(feat_vecs_preps))

    return feat_vecs, new_vectors, feat_vecs_preps


def plot_feature_vectors(vectors_file_name, vector_preps_file_name, prepositions):
    # Retrieve dimensionally reduced vectors for each subj/obj/subj-obj
    label_location = "prep_occurrences/" + vectors_file_name + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve corresponding most dominant preposition for each subj/obj/subj-obj
    label_location = "prep_occurrences/" + vector_preps_file_name + ".p"
    vector_preps = np.asarray(pickle.load(open(label_location, "rb")))

    # # check when IN starts and ends
    # for i in range(len(vector_preps)):
    #     if vector_preps[i] == 26:
    #         print(i)
    #         return i

    print(vectors[0])

    # IN 114974:340327
    # ON 373986:776315

    # # 2-D ######################################################
    # plt.subplot(1, 2, 1)
    # plt.gca().set_aspect('equal', adjustable='box')
    #
    # # plt.plot(vectors[114974:340327, 0], vectors[114974:340327, 1], 'bo', markersize=1)
    # plt.plot(vectors[373986:776315, 0], vectors[373986:776315, 1], 'ro', markersize=1)
    #
    #
    # # Add x, y axis
    # plt.axhline(linewidth=0.5, color='k')
    # plt.axvline(linewidth=0.5, color='k')
    #
    # # Kernel Density Estimator
    # plt.subplot(1, 2, 2)
    # plt.axis([-10, 10, -10, 10])
    #
    # # kde_x_normalized_diff = vectors[114974:340327, 0]
    # # kde_y_normalized_diff = vectors[114974:340327, 1]
    #
    # kde_x_normalized_diff = vectors[373986:776315, 0]
    # kde_y_normalized_diff = vectors[373986:776315, 1]
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
    #
    # plt.colorbar()
    # ###########################
    #
    # plt.show()
    # ############################################################

    # 1D

    # IN
    in_overlap_vecs = vectors[114974:340327, 4]

    # ON
    on_overlap_vecs = vectors[373986:776315, 4]

    bins = np.linspace(0, 1, 100)

    plt.hist(on_overlap_vecs, bins, alpha=0.5, label='on')
    plt.hist(in_overlap_vecs, bins, alpha=0.5, label='in')

    plt.legend(loc='upper right')
    # plt.axis([0, 1, 0, 5000])

    pl.show()


def word_vec_classify(vector_file_name, dominant_prep, classification_method, preps):
    # Retrieve vectors for each word
    label_location = "prep_occurrences/" + vector_file_name + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Retrieve prepositions associated with each word only if reduction_method is 'lda'
    prep_order_location = "prep_occurrences/" + dominant_prep + ".p"
    vector_prepositions = np.asarray(pickle.load(open(prep_order_location, "rb")))

    # Get X and y
    X = np.array(vectors)
    y = np.array(vector_prepositions)

    # Randomly shuffle X and y
    X, y = shuffle(X, y, random_state=0)

    # Number of dimensions to reduce to
    n_dimensions = len(preps) - 1

    # Get priors
    priors_count = []
    total_count = 0.0
    for prep in preps:
        data = get_prep_data(prep)
        priors_count.append(len(data))
        total_count += len(data)
    # print(priors_count)
    for i in range(len(priors_count)):
        priors_count[i] = priors_count[i] / total_count
    # print("total_count", total_count)
    # print("priors_count", priors_count)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Classify
    if classification_method == 'lda':
        clf = LinearDiscriminantAnalysis(n_components=n_dimensions, solver='eigen', priors=np.asarray(priors_count), shrinkage=0.5)
        # clf = LinearDiscriminantAnalysis(n_components=n_dimensions, solver='eigen', shrinkage=0.5)

        scores = cross_val_score(clf, X, y, cv=5)
        print("lda:", scores)
        print("mean:", scores.mean())

        # clf.fit(X_train, y_train)
        #
        # train_count_correct = 0
        # for i in range(len(X_train)):
        #     # print([X_test[i]])
        #     prediction = clf.predict(np.array([X_train[i]]))
        #     if prediction == y_train[i]:
        #         train_count_correct += 1
        # print('train_count_correct', train_count_correct)
        # print('train_total', len(X_train))
        # print('train_percentage:', train_count_correct / len(X_train))
        #
        # test_count_correct = 0
        # for i in range(len(X_test)):
        #     # print([X_test[i]])
        #     prediction = clf.predict(np.array([X_test[i]]))
        #     if prediction == y_test[i]:
        #         test_count_correct += 1
        # print('test_count_correct', test_count_correct)
        # print('test_total', len(X_test))
        # print('test_percentage:', test_count_correct / len(X_test))

    elif classification_method == 'qda':
        clf = QuadraticDiscriminantAnalysis(priors=np.asarray(priors_count))
        # clf = QuadraticDiscriminantAnalysis()

        scores = cross_val_score(clf, X, y, cv=5)
        print("qda:", scores)
        print("mean:", scores.mean())

        # clf.fit(X_train, y_train)
        #
        # train_count_correct = 0
        # for i in range(len(X_train)):
        #     # print([X_train[i]])
        #     prediction = clf.predict(np.array([X_train[i]]))
        #     if prediction == y_train[i]:
        #         train_count_correct += 1
        # print('train_count_correct', train_count_correct)
        # print('train_test_total', len(X_train))
        # print('train_percentage:', train_count_correct / len(X_train))
        #
        # test_count_correct = 0
        # for i in range(len(X_test)):
        #     # print([X_test[i]])
        #     prediction = clf.predict(np.array([X_test[i]]))
        #     if prediction == y_test[i]:
        #         test_count_correct += 1
        # print('test_count_correct', test_count_correct)
        # print('test_total', len(X_test))
        # print('test_percentage:', test_count_correct / len(X_test))


def get_word_vectors_and_feature_vectors(prepositions, model, vocab):
    word_and_feature_vectors = []
    word_and_feature_vectors_prep = []
    for prep in prepositions:
        data = get_prep_data(prep)
        for i in range(len(data)):
            if data.iloc[i, 2] in vocab and data.iloc[i, 9] in vocab:
                concatenated_word_vectors = np.concatenate((np.asarray(model[data.iloc[i, 2]]),
                                                            np.asarray(model[data.iloc[i, 9]])))

                feature_vector = get_feature_vector(data, i)

                word_and_feature_vector = np.concatenate((concatenated_word_vectors, feature_vector))
                word_and_feature_vectors.append(word_and_feature_vector)
                word_and_feature_vectors_prep.append(prepositions.index(prep))

                # print(len(word_and_feature_vector))
        print(prep, "done")

    return word_and_feature_vectors, word_and_feature_vectors_prep


def random_data_sampling(prepositions):
    for prep in prepositions:
        print(prep)
        data = get_prep_data(prep)
        index_list = list(range(0, len(data)))
        random.shuffle(index_list)

        for i in range(0, 400):
            box_images(prep, index_list[i])


def main():
    prepositions = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # Spatial prepositions
    prepositions = ["above", "across", "against", "along", "behind", "beside", "beyond", "in", "on", "under"]

    # prepositions = ['in', 'on']

    # # Plot each preposition by itself
    # for prep in prepositions:
    #     outlier_img_ids = plot_normalized_centre_points(prep) #return outlier_img_ids for when want to check for weird images

    # Plot all prepositions together, each with different colour
    # plot_all(prepositions)

    # # Check for weird images
    # print(prep + " has " + str(len(outlier_img_ids)) + "/" + str(len(data)) + " (" + str(617/13844*100.0) + "%)"
    #       + " weird images")
    # box_images(outlier_img_ids)

    # # Box all images in order of prepositions
    # box_images(prepositions)

    # # Get word vectors for unique obj/subj/subj_obj and their most dominant prep
    #
    # # Get all unique subj, obj, and (subj, obj)
    # unique_subj, unique_obj, unique_subj_obj = get_obj_and_subj(prepositions)
    #
    # # Sort the subj, obj, and (subj, obj) lists
    # unique_subj.sort()
    # unique_obj.sort()
    # unique_subj_obj.sort()
    #
    # # Save unique_obj, unique_subj, unique_subj_obj as pickle files
    # pickle_file(unique_obj, "unique_obj")
    # pickle_file(unique_subj, "unique_subj")
    # pickle_file(unique_subj_obj, "unique_subj_obj")
    #
    # # Fill the tables (inputting 3 at once to reduce number of calls)
    # prep_subj, prep_obj, prep_subj_obj = fill_tables("unique_obj", "unique_subj", "unique_subj_obj", prepositions)
    #
    # # Save prep_obj, prep_subj, prep_subj_obj as pickle files
    # pickle_file(prep_obj, "prep_obj")
    # pickle_file(prep_subj, "prep_subj")
    # pickle_file(prep_subj_obj, "prep_subj_obj")
    #
    # Plot 9 plots. 3 for subj, 3 for obj, 3 for subj - obj. The 3 plots for each subj/obj/subj - obj respectively is
    # using a different dimensionality reduction method - PCA, t-SNE, LDA respectively
    # subj_most_dominant_prep = get_most_dominant_prep('prep_subj')
    # print(len(subj_most_dominant_prep))
    # obj_most_dominant_prep = get_most_dominant_prep('prep_obj')
    # print(len(obj_most_dominant_prep))
    # subj_obj_most_dominant_prep = get_most_dominant_prep('prep_subj_obj')
    # print(len(subj_obj_most_dominant_prep))
    #
    # # Save obj, subj, subj_obj, prep_order as pickle files
    # pickle_file(subj_most_dominant_prep, "subj_most_dominant_prep")
    # print("subj_most_dominant_prep saved")
    # pickle_file(obj_most_dominant_prep, "obj_most_dominant_prep")
    # print("obj_most_dominant_prep saved")
    # pickle_file(subj_obj_most_dominant_prep, "subj_obj_most_dominant_prep")
    # print("subj_obj_most_dominant_prep saved")
    #
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print("word2vec model loaded")
    # word_vectors = model.wv
    # vocab = word_vectors.vocab
    #
    # subj_vec, subj_vec_preps = word_to_vector("unique_subj", "subj_most_dominant_prep", model, vocab)
    # print('len(subj_vec)', len(subj_vec))
    # print('len(subj_vec_preps)', len(subj_vec_preps))
    # print("subj_vec done")
    # obj_vec, obj_vec_preps = word_to_vector("unique_obj", "obj_most_dominant_prep", model, vocab)
    # print('len(obj_vec)', len(obj_vec))
    # print('len(obj_vec_preps)', len(obj_vec_preps))
    # print("obj_vec done")
    # subj_obj_vec, subj_obj_vec_preps = word_to_vector("unique_subj_obj", "subj_obj_most_dominant_prep", model, vocab)
    # print('len(subj_obj_vec)', len(subj_obj_vec))
    # print('len(subj_obj_vec_preps)', len(subj_obj_vec_preps))
    # print("subj_obj_vec done")
    # # Addition vectors
    # subj_obj_add_vec, subj_obj_vec_preps = word_to_vector("unique_subj_obj", "subj_obj_most_dominant_prep", model, vocab)
    # print('len(subj_obj_add_vec)', len(subj_obj_add_vec))
    # # print('len(subj_obj_vec_preps)', len(subj_obj_vec_preps))
    # print("subj_obj_add_vec done")
    #
    # pickle_file(subj_vec, "subj_vec")
    # print("subj_vec saved")
    # pickle_file(obj_vec, "obj_vec")
    # print("obj_vec saved")
    # pickle_file(subj_obj_vec, "subj_obj_vec")
    # print("subj_obj_vec saved")
    # pickle_file(subj_obj_add_vec, "subj_obj_add_vec")
    # print("subj_obj_add_vec saved")
    #
    # pickle_file(subj_vec_preps, "subj_vec_preps")
    # print("subj_vec_preps saved")
    # pickle_file(obj_vec_preps, "obj_vec_preps")
    # print("obj_vec_preps saved")
    # pickle_file(subj_obj_vec_preps, "subj_obj_vec_preps")
    # print("subj_obj_vec_preps saved")

    # Reduce dimensionality for each subj/obj/subj-obj and each pca/tsne/lda
    #
    # subj_vec_reduced_pca = reduce_dimensionality('subj_vec', None, 'pca')
    # print("subj_vec_reduced_pca reduced")
    # pickle_file(subj_vec_reduced_pca, "subj_vec_reduced_pca")
    # print("subj_vec_reduced_pca saved")
    #
    # subj_vec_reduced_tsne = reduce_dimensionality('subj_vec', None, 'tsne')
    # print("subj_vec_reduced_tsne reduced")
    # pickle_file(subj_vec_reduced_tsne, "subj_vec_reduced_tsne")
    # print("subj_vec_reduced_tsne saved")
    #
    # subj_vec_reduced_lda = reduce_dimensionality('subj_vec', 'subj_vec_preps', 'lda')
    # print("subj_vec_reduced_lda reduced")
    # pickle_file(subj_vec_reduced_lda, "subj_vec_reduced_lda")
    # print("subj_vec_reduced_lda saved")
    #
    # obj_vec_reduced_pca = reduce_dimensionality('obj_vec', None, 'pca')
    # print("obj_vec_reduced_pca reduced")
    # pickle_file(obj_vec_reduced_pca, "obj_vec_reduced_pca")
    # print("obj_vec_reduced_pca saved")
    #
    # obj_vec_reduced_tsne = reduce_dimensionality('obj_vec', None, 'tsne')
    # print("obj_vec_reduced_tsne reduced")
    # pickle_file(obj_vec_reduced_tsne, "obj_vec_reduced_tsne")
    # print("obj_vec_reduced_tsne saved")
    #
    # obj_vec_reduced_lda = reduce_dimensionality('obj_vec', 'obj_vec_preps', 'lda')
    # print("obj_vec_reduced_lda reduced")
    # pickle_file(obj_vec_reduced_lda, "obj_vec_reduced_lda")
    # print("obj_vec_reduced_lda saved")
    #
    # subj_obj_vec_reduced_pca = reduce_dimensionality('subj_obj_vec', None, 'pca')
    # print("sbj_obj_vec_reduced_pca reduced")
    # pickle_file(subj_obj_vec_reduced_pca, "subj_obj_vec_reduced_pca")
    # print("subj_obj_vec_reduced_pca saved")
    #
    # subj_obj_vec_reduced_tsne = reduce_dimensionality('subj_obj_vec', None, 'tsne')
    # print("subj_obj_vec_reduced_tsne reduced")
    # pickle_file(subj_obj_vec_reduced_tsne, "subj_obj_vec_reduced_tsne")
    # print("subj_obj_vec_reduced_tsne saved")
    #
    # subj_obj_vec_reduced_lda = reduce_dimensionality('subj_obj_vec', 'subj_obj_vec_preps', 'lda')
    # print("subj_obj_vec_reduced_lda reduced")
    # pickle_file(subj_obj_vec_reduced_lda, "subj_obj_vec_reduced_lda")
    # print("subj_obj_vec_reduced_lda saved")
    #
    # subj_obj_add_vec_reduced_lda = reduce_dimensionality('subj_obj_add_vec', 'subj_obj_vec_preps', 'lda')
    # print("subj_obj_add_vec_reduced_lda reduced")
    # pickle_file(subj_obj_add_vec_reduced_lda, "subj_obj_add_vec_reduced_lda")
    # print("subj_obj_add_vec_reduced_lda saved")

    # Plot all dimensionality reduced plots
    #
    # plot_vectors("subj_vec_reduced_pca", "subj_vec_preps", "Subjects: PCA", "2", prepositions)
    # plot_vectors("obj_vec_reduced_pca", "obj_vec_preps", "Objects: PCA", "2", prepositions)
    # plot_vectors("subj_obj_vec_reduced_pca", "subj_obj_vec_preps", "Subjects - Objects: PCA", "2", prepositions)
    #
    # plot_vectors("subj_vec_reduced_tsne", "subj_vec_preps", "Subjects: t-SNE", "2", prepositions)
    # plot_vectors("obj_vec_reduced_tsne", "obj_vec_preps", "Objects: t-SNE", "2", prepositions)
    # plot_vectors("subj_obj_vec_reduced_tsne", "subj_obj_vec_preps", "Subjects - Objects: t-SNE", "2", prepositions)
    #
    # plot_vectors("subj_vec_reduced_lda", "subj_vec_preps", "Subjects: LDA", "2", prepositions)
    # plot_vectors("obj_vec_reduced_lda", "obj_vec_preps", "Objects: LDA", "2", prepositions)
    # plot_vectors("subj_obj_vec_reduced_lda", "subj_obj_vec_preps", "Subjects - Objects: LDA", "2", prepositions)
    # plot_vectors("subj_obj_add_vec_reduced_lda", "subj_obj_vec_preps", "Subjects + Objects: LDA", "2", prepositions)
    #
    # plot_vectors("subj_vec_reduced_pca", "subj_vec_preps", "Subjects: PCA", "3", prepositions)
    # plot_vectors("obj_vec_reduced_pca", "obj_vec_preps", "Objects: PCA", "3", prepositions)
    # plot_vectors("subj_obj_vec_reduced_pca", "subj_obj_vec_preps", "Subjects - Objects: PCA", "3", prepositions)
    #
    # plot_vectors("subj_vec_reduced_tsne", "subj_vec_preps", "Subjects: t-SNE", "3", prepositions)
    # plot_vectors("obj_vec_reduced_tsne", "obj_vec_preps", "Objects: t-SNE", "3", prepositions)
    # plot_vectors("subj_obj_vec_reduced_tsne", "subj_obj_vec_preps", "Subjects - Objects: t-SNE", "3", prepositions)
    #
    # plot_vectors("subj_vec_reduced_lda", "subj_vec_preps", "Subjects: LDA", "3", prepositions)
    # plot_vectors("obj_vec_reduced_lda", "obj_vec_preps", "Objects: LDA", "3", prepositions)
    # plot_vectors("subj_obj_vec_reduced_lda", "subj_obj_vec_preps", "Subjects - Objects: LDA", "3", prepositions)


    # # Bounding Box Method
    #
    # Get longest edge of all subjs and objs of all prepositions
    # longest_separation, longest_separation_prep, longest_separation_prep_index, longest_separation_img_id \
    #     = find_longest_separation(prepositions)
    # longest_edge = 1281
    # longest_separation = 1126.0
    # print('find_longest(prepositions) done')
    # print('longest_separation', longest_separation)
    # print('longest_separation_prep', longest_separation_prep)
    # print('longest_separation_prep_index', longest_separation_prep_index)
    # print('longest_separation_img_id', longest_separation_img_id)
    #
    # decrease_factor = 0.5 / longest_edge
    # normalized_longest_separation = longest_separation * decrease_factor
    # print('decrease_factor', decrease_factor)
    # print('normalized_longest_separation', normalized_longest_separation)
    #
    # for prep in prepositions:
    #     get_bounding_box_vectors(prep, decrease_factor, normalized_longest_separation)
    # print('longest_edge', longest_edge)
    #
    # for prep in prepositions:
    #     get_normalized_bounding_box_vectors(prep, 1)
    # print('longest_edge', longest_edge)


    # For presentation
    #
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print("word2vec model loaded")
    # # word_vectors = model.wv
    # # vocab = word_vectors.vocab
    #
    # print("cup", model["cup"])
    # print(len(model["cup"]))
    # print("table", model["table"])
    # print(len(model["table"]))
    # # subj_vec, subj_vec_preps = word_to_vector("subj", "subj_most_dominant_prep", model, vocab)

    # # Plot individual normalized points ##################
    #
    # prep = "above"
    # data = get_prep_data(prep)
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
    # plt.title(prep)
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
    # plt.savefig("../Presentation-Images/" + prep + "_" + str(i) + "_centroid.png", dpi=300)
    #
    # plt.show()
    #
    # ##############################################

    # for prep in prepositions:
    #     feature_vectors = get_feature_vectors(prep)
    #     pickle_file(feature_vectors, prep + "_feat_vecs")
    #     print(prep + " feature vectors saved")
    #
    # feature_vectors, dim_reduce_feature_vectors, feature_vectors_preps = dim_reduc_feat_vecs(prepositions)
    # pickle_file(feature_vectors, "feature_vectors")
    # pickle_file(dim_reduce_feature_vectors, "dim_reduce_feature_vectors")
    # pickle_file(feature_vectors_preps, "feature_vectors_preps")

    # plot_vectors("dim_reduce_feature_vectors", "dim_reduce_feature_vectors_preps", 'hi', '2', prepositions)
    # plot_feature_vectors("dim_reduce_feature_vectors", "feature_vectors_preps", prepositions)
    # plot_feature_vectors("feature_vectors", "feature_vectors_preps", prepositions)
    #
    #
    # in_feat_vec_location = "prep_occurrences/" + prepositions[20] + "_feat_vecs.p"
    # in_feat_vecs = np.asarray(pickle.load(open(in_feat_vec_location, "rb")))
    # in_overlap_vecs = in_feat_vecs[:, 4]
    #
    # on_feat_vec_location = "prep_occurrences/" + prepositions[25] + "_feat_vecs.p"
    # on_feat_vecs = np.asarray(pickle.load(open(on_feat_vec_location, "rb")))
    # on_overlap_vecs = on_feat_vecs[:, 4]
    #
    # bins = np.linspace(0, 1, 100)
    #
    # plt.hist(on_overlap_vecs, bins, alpha=0.5, label='on')
    # plt.hist(in_overlap_vecs, bins, alpha=0.5, label='in')
    #
    # plt.legend(loc='upper right')
    # plt.axis([0, 1, 0, 10000])
    #
    # pl.show()

    # Classifiers ######################

    # # All prepositions
    #
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print("word2vec model loaded")
    word_vectors = model.wv
    vocab = word_vectors.vocab

    word_and_feature_vectors, word_and_feature_vectors_prep = get_word_vectors_and_feature_vectors(prepositions, model, vocab)

    pickle_file("word_and_feature_vectors", word_and_feature_vectors)
    pickle_file("word_and_feature_vectors_prep", word_and_feature_vectors_prep)

    word_vec_classify('word_and_feature_vectors', 'word_and_feature_vectors_prep', 'lda', prepositions)
    word_vec_classify('word_and_feature_vectors', 'word_and_feature_vectors_prep', 'qda', prepositions)

    # Using concat(subj, obj), LDA and QDA accuracy are on avg 0.51161 and 0.63085897
    # word_vec_classify('subj_obj_vec', 'subj_obj_vec_preps', 'lda', prepositions)
    # word_vec_classify('subj_obj_vec', 'subj_obj_vec_preps', 'qda', prepositions)

    # JUST IN and ON

    # subj_obj_most_dominant_prep = get_most_dominant_prep('in_on_prep_subj_obj')
    # print(len(subj_obj_most_dominant_prep))
    #
    # pickle_file(subj_obj_most_dominant_prep, "in_on_subj_obj_most_dominant_prep")
    # print("in_on_subj_obj_most_dominant_prep saved")
    #
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print("word2vec model loaded")
    # word_vectors = model.wv
    # vocab = word_vectors.vocab
    #
    # subj_obj_vec, subj_obj_vec_preps = word_to_vector("in_on_subj_obj", "in_on_subj_obj_most_dominant_prep", model, vocab)
    # print('len(in_on_subj_obj_vec)', len(subj_obj_vec))
    # print('len(in_on_subj_obj_vec_preps)', len(subj_obj_vec_preps))
    # print("in_on_subj_obj_vec done")
    # subj_obj_add_vec, subj_obj_vec_preps = word_to_vector("in_on_subj_obj", "in_on_subj_obj_most_dominant_prep", model, vocab)
    # print('len(in_on_subj_obj_add_vec)', len(subj_obj_add_vec))
    # # print('len(subj_obj_vec_preps)', len(subj_obj_vec_preps))
    # print("in_on_subj_obj_add_vec done")
    #
    # pickle_file(subj_obj_vec, "in_on_subj_obj_vec")
    # print("in_on_subj_obj_vec saved")
    # pickle_file(subj_obj_add_vec, "in_on_subj_obj_add_vec")
    # print("in_on_subj_obj_add_vec saved")
    #
    # pickle_file(subj_obj_vec_preps, "in_on_subj_obj_vec_preps")
    # print("in_on_subj_obj_vec_preps saved")
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
    # subj_vec_reduced_lda = reduce_dimensionality('subj_vec', 'subj_vec_preps', 'lda')
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
    # obj_vec_reduced_lda = reduce_dimensionality('obj_vec', 'obj_vec_preps', 'lda')
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
    # subj_obj_vec_reduced_lda = reduce_dimensionality('subj_obj_vec', 'subj_obj_vec_preps', 'lda')
    # print("subj_obj_vec_reduced_lda reduced")
    # pickle_file(subj_obj_vec_reduced_lda, "subj_obj_vec_reduced_lda")
    # print("subj_obj_vec_reduced_lda saved")
    #
    # subj_obj_add_vec_reduced_lda = reduce_dimensionality('subj_obj_add_vec', 'subj_obj_vec_preps', 'lda')
    # print("subj_obj_add_vec_reduced_lda reduced")
    # pickle_file(subj_obj_add_vec_reduced_lda, "subj_obj_add_vec_reduced_lda")
    # print("subj_obj_add_vec_reduced_lda saved")

    # word_vec_classify('in_on_subj_obj_vec', 'in_on_subj_obj_vec_preps', 'lda', prepositions)
    # word_vec_classify('subj_obj_vec', 'subj_obj_vec_preps', 'qda', prepositions)


    # # Sample high quality data
    # random_data_sampling(prepositions)


if __name__ == "__main__":
    main()
