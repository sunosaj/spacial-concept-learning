import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import csv
from scipy.stats import kde
import pandas as pd

# img_dict = {}


def get_data(pred):
    file_address = 'new_pred/data_pred_' + pred + '.csv'
    print(file_address)
    data = pd.read_csv(file_address)
    return data


# def to_dictionary(data):
#     # Convert data to a dictionary with image_id to be key and rest of each row to be value
#     for i in range(len(data)):
#         img_dict[int(data[i][0])] = data[i][1:]


def plot(pred, data):
    # Get obj and subj centre points and normalize diffs
    x_normalized_diff = []
    y_normalized_diff = []

    # Weird images
    outlier_relationships = []

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
    # plt.savefig("new_pred_plot/" + pred + "_plot.eps", dpi=40)

    plt.show()

    return outlier_relationships


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


def main():
    predicates = ["on"]
    # ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
    #  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
    #  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
    #  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
    #  "within", "without"]

    for pred in predicates:
        data = get_data(pred)
        outlier_img_ids = plot(pred, data) #return outlier_img_ids for when want to check for weird images


    # # Check for weird images
    # print(pred + " has " + str(len(outlier_img_ids)) + "/" + str(len(data)) + " (" + str(617/13844*100.0) + "%)"
    #       + " weird images")
    # box_images(outlier_img_ids)

    # Check for images with numbers as labels for objects and subjects
    # print(data.iloc[0, 1])
    # img_ids = [285624]
    # box_images(img_ids, data)


if __name__ == "__main__":
    main()
