import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import csv
from scipy.stats import kde
from matplotlib.backends.backend_pdf import PdfPages
import math

img_dict = {}
# pp = PdfPages('pred_plots.pdf')


def get_data(pred):
    # Open data file for pred and get its data as a 2d list
    file_address = 'new_pred/data_pred_' + pred + '.csv'
    data = []

    with open(file_address, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data[1:]


def to_dictionary(data):
    # Convert data to a dictionary with image_id to be key and rest of each row to be value
    for i in range(len(data)):
        img_dict[int(data[i][0])] = data[i][1:]


def plot(pred, data):
    # Get obj and subj centre points and normalize diffs
    x_normalized_diff = []
    y_normalized_diff = []

    # Weird images
    weird_img_ids = []

    for row in data:
        obj_centre_x = float(row[5]) + (float(row[4]) / 2.0)
        obj_centre_y = float(row[6]) + (float(row[3]) / 2.0)
        subj_centre_x = float(row[12]) + (float(row[11]) / 2.0)
        subj_centre_y = float(row[13]) + (float(row[10]) / 2.0)
        x_normalized_diff.append((subj_centre_x - obj_centre_x) / float(row[15]))
        y_normalized_diff.append((obj_centre_y - subj_centre_y) / float(row[16]))

        # For specific predicates #
        if pred == 'above':
            if y_normalized_diff[-1] < 0:
                weird_img_ids.append(int(row[0]))
        elif pred == 'below':
            if y_normalized_diff[-1] > 0:
                weird_img_ids.append(int(row[0]))
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

    ######################################################

    # # Kernel Density Estimator Method 1
    # xmin = np.asarray(x_normalized_diff).min()
    # xmax = np.asarray(x_normalized_diff).max()
    # ymin = np.asarray(y_normalized_diff).min()
    # ymax = np.asarray(y_normalized_diff).max()
    #
    # # Perform a kernel density estimate on the data:
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([np.asarray(x_normalized_diff), np.asarray(y_normalized_diff)])
    # kernel = stats.gaussian_kde(values)
    # Z = np.reshape(kernel(positions).T, X.shape)
    #
    # # # Plot the results (original):
    # # plt.subplot(1, 2, 2)
    # # plt.axis([-1, 1, -1, 1])
    #
    # f, ax = plt.subplots()
    # im = ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    # # ax.plot(np.asarray(x_normalized_diff), np.asarray(y_normalized_diff), 'k.', markersize=2)
    #
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # f.colorbar(im)
    # plt.plot([-1, 1], [0, 0], 'k-', lw=2)
    # plt.plot([0, 0], [-1, 1], 'k-', lw=2)
    # plt.title(pred)

    ######################################################

    #Kernel Density Estimator Method 2###
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
    plt.title(pred)
    plt.colorbar()

    ##############################

    # #Kernel Density Estimator Method 3#
    # xmin = np.asarray(x_normalized_diff).min()
    # xmax = np.asarray(x_normalized_diff).max()
    # ymin = np.asarray(y_normalized_diff).min()
    # ymax = np.asarray(y_normalized_diff).max()
    #
    # # Perform a kernel density estimate on the data:
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # values = np.vstack([np.asarray(x_normalized_diff), np.asarray(y_normalized_diff)])
    # kernel = stats.gaussian_kde(values)
    # Z = np.reshape(kernel(positions).T, X.shape)
    #
    # # # Plot the results (original):
    # # plt.subplot(1, 2, 2)
    # # plt.axis([-1, 1, -1, 1])
    #
    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(121)
    # ax1.set_xlim([-1, 1])
    # ax1.set_ylim([-1, 1])
    # plt.title(pred)
    # ax1.plot(x_normalized_diff, y_normalized_diff, 'ro', markersize=1)
    # plt.axhline(linewidth=0.5, color='k')
    # plt.axvline(linewidth=0.5, color='k')
    # # plt.gca().set_aspect('equal', adjustable='box')
    #
    # ax2 = fig.add_subplot(122)
    #
    # # f, ax = plt.subplots()
    #
    # im = ax2.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    #
    # ax2.set_xlim([-1, 1])
    # ax2.set_ylim([-1, 1])
    # plt.axhline(linewidth=0.5, color='k')
    # plt.axvline(linewidth=0.5, color='k')
    # plt.title(pred)
    # fig.colorbar(im)

    ###################################

    # # Save plots
    plt.savefig("new_pred_plot/" + pred + "_plot.pdf")

    # plt.show()

    return weird_img_ids


def box_images(image_ids):
    for image_id in image_ids:
        # Open image
        image_address = 'images/' + str(image_id) + '.jpg'
        im = np.array(Image.open(image_address), dtype=np.uint8)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        # Get image info
        info = img_dict[image_id]

        # Create a Rectangle patch
        obj_box = patches.Rectangle((int(info[4]), int(info[5])), int(info[3]), int(info[2]), linewidth=1,
                                    edgecolor='r', facecolor='none')
        subj_box = patches.Rectangle((int(info[11]), int(info[12])), int(info[10]), int(info[9]), linewidth=1,
                                     edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(obj_box)
        ax.add_patch(subj_box)

        # Label obj and subj
        plt.annotate("obj: " + info[1], (int(info[4]), int(info[5])))
        plt.annotate("subj: " + info[8], (int(info[11]), int(info[12])))

        plt.title(image_id)
        plt.show()


def main():
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "nearby", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # for pred in predicates:
    pred = "without"
    data = get_data(pred)
    to_dictionary(data)
    weird_img_ids = plot(pred, data) #return weird_img_ids for when want to check for weird images


    # # Check for weird images
    # print(pred + " has " + str(len(weird_img_ids)) + "/" + str(len(data)) + " (" + str(617/13844*100.0) + "%)"
    #       + " weird images")
    # box_images(weird_img_ids)


if __name__ == "__main__":
    main()
