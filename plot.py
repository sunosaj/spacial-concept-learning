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


def plot(pred):
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


def box_images(image_ids, data):
    for image_id in image_ids:
        # Open image
        image_address = 'images/' + str(image_id) + '.jpg'
        im = np.array(Image.open(image_address), dtype=np.uint8)

        # Box all relationships for this image_id
        for i in range(len(data)):
            if data.iloc[i, 0] == image_id:
                # Create figure and axes
                fig, ax = plt.subplots(1)

                # Display the image
                ax.imshow(im)

                # Create a Rectangle patch
                obj_box = patches.Rectangle((int(data.iloc[i, 5]), int(data.iloc[i, 6])), int(data.iloc[i, 4]),
                                            int(data.iloc[i, 3]), linewidth=1, edgecolor='r', facecolor='none')
                subj_box = patches.Rectangle((int(data.iloc[i, 12]), int(data.iloc[i, 13])), int(data.iloc[i, 11]),
                                             int(data.iloc[i, 10]), linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(obj_box)
                ax.add_patch(subj_box)

                # Label obj and subj
                plt.annotate("obj: " + data.iloc[i, 2], (int(data.iloc[i, 5]), int(data.iloc[i, 6])))
                plt.annotate("subj: " + data.iloc[i, 9], (int(data.iloc[i, 12]), int(data.iloc[i, 13])))

                plt.title(image_id)
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
        if isinstance(words[i], tuple):
            if words[i][0] in vocab and words[i][1] in vocab:
                word_vectors.append(model[words[i][0]] - model[words[i][1]])
                new_pred_order.append(preds[i])
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

    # Retrieve predicates associated with each word only if reduction_method is 'fda'
    vector_predicates = None
    if dominant_pred is not None:
        pred_order_location = "pred_occurrences/" + dominant_pred + ".p"
        vector_predicates = np.asarray(pickle.load(open(pred_order_location, "rb")))

    # print('len(vectors)', len(vectors))
    # print('len(vector_predicates)', len(vector_predicates))

    # Number of dimensions to reduce to
    n_dimensions = len(vectors[0]) - 1

    if reduction_method == 'pca':
        pca = PCA(n_components=n_dimensions)
        pca.fit(vectors)
        new_vectors = pca.transform(vectors)

    elif reduction_method == 'tsne':
        vectors_embedded = TSNE(n_components=3).fit_transform(vectors)
        new_vectors = vectors_embedded

    elif reduction_method == 'fda':
        clf = LinearDiscriminantAnalysis(n_components=n_dimensions)
        clf.fit(vectors, vector_predicates)
        new_vectors = clf.transform(vectors)

    return new_vectors


def plot_vectors(vectors_file_name, title, n_dimensions):
    # Retrieve labels for each row in table
    label_location = "pred_occurrences/" + vectors_file_name + ".p"
    vectors = np.asarray(pickle.load(open(label_location, "rb")))

    # Plot 2-d
    if n_dimensions == '2':
        plt.figure(figsize=(8, 8))
        plt.plot(vectors[:, 0], vectors[:, 1], 'ro', markersize=0.5)
        plt.title(title + ' 2d')
        plt.savefig("word_vector_plot/" + vectors_file_name + "_plot_2d.pdf")

    # Plot 3-d
    elif n_dimensions == '3':
        # plt.plot(vectors[:3, 0], vectors[:3, 1], 'ro', markersize=1)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=0.1, c='b')
        plt.title(title + ' 3d')
        plt.savefig("word_vector_plot/" + vectors_file_name + "_plot_3d.pdf")

    plt.show()


def main():
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # # Plot each predicate by itself
    # for pred in predicates:
    #     outlier_img_ids = plot(pred) #return outlier_img_ids for when want to check for weird images

    # Plot all predicates together, each with different colour
    # plot_all(predicates)

    # # Check for weird images
    # print(pred + " has " + str(len(outlier_img_ids)) + "/" + str(len(data)) + " (" + str(617/13844*100.0) + "%)"
    #       + " weird images")
    # box_images(outlier_img_ids)

    # Check for images with numbers as labels for objects and subjects
    # print(data.iloc[0, 1])
    # img_ids = [285624]
    # box_images(img_ids, data)

    # # Dimensionality Reduction
    #
    # # Plot 9 plots. 3 for subj, 3 for obj, 3 for subj - obj. The 3 plots for each subj/obj/subj - obj respectively is
    # # using a different dimensionality reduction method - PCA, t-SNE, FDA respectively
    # subj_most_dominant_pred = get_most_dominant_pred('prep_subj')
    # obj_most_dominant_pred = get_most_dominant_pred('prep_obj')
    # subj_obj_most_dominant_pred = get_most_dominant_pred('prep_subj_obj')
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

    # Reduce dimensionality for each subj/obj/subj-obj and each pca/tsne/fda

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
    # subj_vec_reduced_fda = reduce_dimensionality('subj_vec', 'subj_vec_preds', 'fda')
    # print("subj_vec_reduced_fda reduced")
    # pickle_file(subj_vec_reduced_fda, "subj_vec_reduced_fda")
    # print("subj_vec_reduced_fda saved")
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
    # obj_vec_reduced_fda = reduce_dimensionality('obj_vec', 'obj_vec_preds', 'fda')
    # print("obj_vec_reduced_fda reduced")
    # pickle_file(obj_vec_reduced_fda, "obj_vec_reduced_fda")
    # print("obj_vec_reduced_fda saved")
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
    # subj_obj_vec_reduced_fda = reduce_dimensionality('subj_obj_vec', 'subj_obj_vec_preds', 'fda')
    # print("subj_obj_vec_reduced_fda reduced")
    # pickle_file(subj_obj_vec_reduced_fda, "subj_obj_vec_reduced_fda")
    # print("subj_obj_vec_reduced_fda saved")


    # Plot all dimensionality reduced plots

    plot_vectors("subj_vec_reduced_pca", "Subjects: PCA", "2")
    plot_vectors("obj_vec_reduced_pca", "Objects: PCA", "2")
    plot_vectors("subj_obj_vec_reduced_pca", "Subjects - Objects: PCA", "2")

    plot_vectors("subj_vec_reduced_tsne", "Subjects: t-SNE", "2")
    plot_vectors("obj_vec_reduced_tsne", "Objects: t-SNE", "2")
    plot_vectors("subj_obj_vec_reduced_tsne", "Subjects - Objects: t-SNE", "2")

    plot_vectors("subj_vec_reduced_fda", "Subjects: FDA", "2")
    plot_vectors("obj_vec_reduced_fda", "Objects: FDA", "2")
    plot_vectors("subj_obj_vec_reduced_fda", "Subjects - Objects: FDA", "2")
    #
    # plot_vectors("subj_vec_reduced_pca", "Subjects: PCA", "3")
    # plot_vectors("obj_vec_reduced_pca", "Objects: PCA", "3")
    # plot_vectors("subj_obj_vec_reduced_pca", "Subjects - Objects: PCA", "3")
    #
    # plot_vectors("subj_vec_reduced_tsne", "Subjects: t-SNE", "3")
    # plot_vectors("obj_vec_reduced_tsne", "Objects: t-SNE", "3")
    # plot_vectors("subj_obj_vec_reduced_tsne", "Subjects - Objects: t-SNE", "3")
    #
    # plot_vectors("subj_vec_reduced_fda", "Subjects: FDA", "3")
    # plot_vectors("obj_vec_reduced_fda", "Objects: FDA", "3")
    # plot_vectors("subj_obj_vec_reduced_fda", "Subjects - Objects: FDA", "3")


if __name__ == "__main__":
    main()
