import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import csv

img_dict = {}


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
    plt.title(pred)
    plt.plot(x_normalized_diff, y_normalized_diff, 'ro')
    plt.axis([-1, 1, -1, 1])

    # Add x, y axis
    plt.plot([-1, 1], [0, 0], 'k-', lw=2)
    plt.plot([0, 0], [-1, 1], 'k-', lw=2)

    # plt.show()

    # Kernel Density Estimator
    xmin = np.asarray(x_normalized_diff).min()
    xmax = np.asarray(x_normalized_diff).max()
    ymin = np.asarray(y_normalized_diff).min()
    ymax = np.asarray(y_normalized_diff).max()

    # Perform a kernel density estimate on the data:
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([np.asarray(x_normalized_diff), np.asarray(y_normalized_diff)])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Plot the results:
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    # ax.plot(np.asarray(x_normalized_diff), np.asarray(y_normalized_diff), 'k.', markersize=2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.show()

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
    pred = "below"

    data = get_data(pred)
    to_dictionary(data)
    weird_img_ids = plot(pred, data)
    print(pred + " has " + str(len(weird_img_ids)) + "/" + str(len(data)) + " (" + str(617/13844*100.0) + "%)"
          + " weird images")
    box_images(weird_img_ids)

    # plot("below")
    # plot("on")


if __name__ == "__main__":
    main()
