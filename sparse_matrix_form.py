import pickle
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd


subj = []
obj = []
subj_and_obj = []


def get_data(pred):
    file_address = 'new_pred/data_pred_' + pred + '.csv'
    dataset = pd.read_csv(file_address)
    return dataset


def get_obj_and_subj(predicates):
    # Loop through data rows and add new subjects and objects to subjects and objects unique list
    for pred in predicates:
        data = get_data(pred)
        for i in range(len(data)):
            if data.iloc[i, 2] not in obj:
                # print("data.iloc[i, 2]:", data.iloc[i, 2])
                obj.append(data.iloc[i, 2])
            if data.iloc[i, 9] not in subj:
                # print("data.iloc[i, 9]:", data.iloc[i, 9])
                subj.append(data.iloc[i, 9])
            if (data.iloc[i, 9], data.iloc[i, 2]) not in subj_and_obj:
                # print("data.iloc[i, 9], data.iloc[i, 2]:", data.iloc[i,9], ",", data.iloc[i,2])
                subj_and_obj.append((data.iloc[i, 9], data.iloc[i, 2]))
        print("get_obj_and_subj:", pred, "done")


def fill_tables(prep_obj, prep_subj, prep_subj_obj, predicates):
    for pred in predicates:
        data = get_data(pred)
        for i in range(len(data)):
            prep_obj[obj.index(data.iloc[i, 2]), predicates.index(pred)] += 1
            prep_subj[subj.index(data.iloc[i, 9]), predicates.index(pred)] += 1
            prep_subj_obj[subj_and_obj.index((data.iloc[i, 9], data.iloc[i, 2])), predicates.index(pred)] += 1
        print("fill_tables:", pred, "done")


def pickle_file(table, name):
    # https://wiki.python.org/moin/UsingPickle
    file_name = "pred_occurrences/" + name + ".p"
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(table, fp)


def to_sparse_matrix(table):
    # # Doing it this way messes up the sparse matrix generated afterwards
    # file_name = "pred_occurrences/" + table + ".txt"
    # with open(table, "rb") as fp:  # Unpickling
    #     b = pickle.load(fp)
    # print("to_sparse_matrix", b)

    # # https://wiki.python.org/moin/UsingPickle
    # loaded_table = pickle.load(open(table, "rb"))
    # print("loaded_table")
    # print(loaded_table)
    # # Convert table to sparse matrix form
    # # https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix
    # return sparse.csr_matrix(loaded_table)

    return sparse.csr_matrix(table)


def to_pie_graph(table_name, label_names, predicates):
    # If retrieve table from pickled file
    # https://wiki.python.org/moin/UsingPickle
    file_name = "pred_occurrences/" + table_name + ".p"
    table = pickle.load(open(file_name, "rb"))

    # If not retrieve table from pickled file
    # https://pythonspot.com/matplotlib-pie-chart/
    for i in range(len(table)):
        row_sum = int(sum(table[i]))
        labels = []
        for j in range(len(predicates)):
            label = predicates[j] + ": " + str(int(table[i][j])) + "/" + str(row_sum) + \
                    " (" + str(int(table[i][j] / row_sum * 100)) + "%)"
            labels.append(label)

        row = table[i]

        # Only keep predicates that occur at least once for this subj/obj/(subj, obj)
        np_row = np.array(row)
        np_labels = np.array(labels)
        nonzero_labels = np_labels[np.flatnonzero(np_row)]
        nonzero_row = np_row[np.flatnonzero(np_row)]

        # Create pie chart
        label_name = label_names[i]
        plt.figure(figsize=(10, 5))
        plt.title(label_name)
        patches, texts = plt.pie(nonzero_row, startangle=90, counterclock=False)
        plt.legend(patches, nonzero_labels, loc="best")

        # Keep the pie charts circular, not oval
        plt.axis('equal')

        plt.show()


def main():
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    predicates_with_errors = ["around", "at", "in", "under"]

    predicates = ["about", "above", "across", "after", "against"]

    # Get all unique subj, obj, and (subj, obj)
    get_obj_and_subj(predicates)

    # Sort the subj, obj, and (subj, obj) lists
    subj.sort()
    obj.sort()
    subj_and_obj.sort()

    # Initialize subj, obj, and (subj, obj) tables
    prep_subj = np.zeros((len(subj), len(predicates)))
    prep_obj = np.zeros((len(obj), len(predicates)))
    prep_subj_obj = np.zeros((len(subj_and_obj), len(predicates)))

    # Fill the tables (inputting 3 at once to save runtime)
    fill_tables(prep_obj, prep_subj, prep_subj_obj, predicates)

    # Save prep_obj, prep_subj, prep_subj_obj as pickle files
    pickle_file(prep_obj, "prep_obj")
    pickle_file(prep_subj, "prep_subj")
    pickle_file(prep_subj_obj, "prep_subj_obj")

    # prep_obj = to_sparse_matrix("prep_obj")
    # prep_subj = to_sparse_matrix("prep_subj")
    # prep_subj_obj = to_sparse_matrix("prep_subj_obj")

    # # Generate Pie Graphs for each table subj, obj, and (subj, obj)
    # to_pie_graph(prep_obj, obj, predicates)
    # to_pie_graph(prep_subj, subj, predicates)
    # to_pie_graph(prep_subj_obj, subj_and_obj, predicates)

    # Generate Pie Graphs for each table subj, obj, and (subj, obj) (Retrieve tables from pickled files)
    to_pie_graph("prep_obj", obj, predicates)
    to_pie_graph("prep_subj", subj, predicates)
    to_pie_graph("prep_subj_obj", subj_and_obj, predicates)


if __name__ == "__main__":
    main()
