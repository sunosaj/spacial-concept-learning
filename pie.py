import pickle
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

subj = []
obj = []
subj_obj = []


def get_data(pred):
    file_address = 'new_pred/data_pred_' + pred + '.csv'
    dataset = pd.read_csv(file_address, encoding="ISO-8859-1")
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
            if (data.iloc[i, 9], data.iloc[i, 2]) not in subj_obj:
                # print("data.iloc[i, 9], data.iloc[i, 2]:", data.iloc[i,9], ",", data.iloc[i,2])
                subj_obj.append((data.iloc[i, 9], data.iloc[i, 2]))
        print("get_obj_and_subj:", pred, "done")


def fill_tables(prep_obj, prep_subj, prep_subj_obj, predicates):
    for pred in predicates:
        data = get_data(pred)
        for i in range(len(data)):
            prep_obj[obj.index(data.iloc[i, 2]), predicates.index(pred)] += 1
            prep_subj[subj.index(data.iloc[i, 9]), predicates.index(pred)] += 1
            prep_subj_obj[subj_obj.index((data.iloc[i, 9], data.iloc[i, 2])), predicates.index(pred)] += 1
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


def to_pie_graph(table_name, predicates):
    # If retrieve table from pickled file
    # https://wiki.python.org/moin/UsingPickle
    table_location = "pred_occurrences/prep_" + table_name + ".p"
    table = pickle.load(open(table_location, "rb"))

    # Retrieve labels for each row in table
    label_location = "pred_occurrences/" + table_name + ".p"
    label_names = pickle.load(open(label_location, "rb"))

    for i in range(len(table)):
        row_sum = int(sum(table[i]))

        # Only plot obj/subj/(subj, obj) that uses 1000+ predicates
        if row_sum >= 1000:
            legend_labels = []
            for j in range(len(predicates)):
                legend_label = predicates[j] + ": " + str(int(table[i][j])) + "/" + str(row_sum) + \
                        " (" + str("{0:.2f}".format(table[i][j] / row_sum * 100)) + "%)"
                legend_labels.append(legend_label)

            row = table[i]

            # Only keep predicates that occur at least once for this subj/obj/(subj, obj), and
            # sort the predicate occurrences from most used to least used
            np_row = np.array(row)
            np_labels = np.array(legend_labels)
            count_nonzero = np.count_nonzero(np_row)
            nonzero_labels = np_labels[(np.argsort(np_row))[-1:-count_nonzero:-1]]
            nonzero_row = (np.sort(np_row))[-1:-count_nonzero:-1]

            # Create pie chart
            if table_name == "subj_obj":
                label_name = "(subj, obj): (" + label_names[i][0] + ", " + label_names[i][1] + ")"
            else:
                label_name = table_name + ": " + str(label_names[i])
            plt.figure(figsize=(14, 7))
            plt.title(label_name)
            patches, texts = plt.pie(nonzero_row, startangle=90, counterclock=False)
            plt.legend(patches, nonzero_labels, title="Predicates", loc="best")

            # Keep the pie charts circular, not oval
            plt.axis('equal')

            print("label_names[" + str(i) + "]: " + str(label_names[i]))

            # Save image
            if table_name == "subj_obj":
                plt.savefig("pie_chart/" + table_name + "/" + label_names[i][0] + "_" + label_names[i][1] +
                            "_pie_chart.pdf")
            else:
                plt.savefig("pie_chart/" + table_name + "/" + label_names[i] + "_pie_chart.pdf")

            # plt.show()


def main():
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # # Get all unique subj, obj, and (subj, obj)
    # get_obj_and_subj(predicates)

    # # Sort the subj, obj, and (subj, obj) lists
    # subj.sort()
    # obj.sort()
    # subj_obj.sort()

    # # Initialize subj, obj, and (subj, obj) tables
    # prep_subj = np.zeros((len(subj), len(predicates)))
    # prep_obj = np.zeros((len(obj), len(predicates)))
    # prep_subj_obj = np.zeros((len(subj_and_obj), len(predicates)))

    # # Fill the tables (inputting 3 at once to save runtime)
    # fill_tables(prep_obj, prep_subj, prep_subj_obj, predicates)

    # # Save obj, subj, subj_obj as pickle files
    # pickle_file(obj, "obj")
    # pickle_file(subj, "subj")
    # pickle_file(subj_obj, "subj_obj")

    # # Save prep_obj, prep_subj, prep_subj_obj as pickle files
    # pickle_file(prep_obj, "prep_obj")
    # pickle_file(prep_subj, "prep_subj")
    # pickle_file(prep_subj_obj, "prep_subj_obj")
    #
    # # prep_obj = to_sparse_matrix("prep_obj")
    # # prep_subj = to_sparse_matrix("prep_subj")
    # # prep_subj_obj = to_sparse_matrix("prep_subj_obj")
    #
    # # # Generate Pie Graphs for each table subj, obj, and (subj, obj)
    # # to_pie_graph(prep_obj, obj, predicates)
    # # to_pie_graph(prep_subj, subj, predicates)
    # # to_pie_graph(prep_subj_obj, subj_and_obj, predicates)

    # Generate Pie Graphs for each table subj, obj, and (subj, obj) (Retrieve tables from pickled files)
    to_pie_graph("subj_obj", predicates)
    to_pie_graph("subj", predicates)
    to_pie_graph("obj", predicates)


if __name__ == "__main__":
    main()
