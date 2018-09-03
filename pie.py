import pickle
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd


def get_data(pred):
    file_address = 'new_pred/data_pred_' + pred + '.csv'
    dataset = pd.read_csv(file_address, encoding="ISO-8859-1")
    return dataset


def get_obj_and_subj(predicates):
    subj = []
    obj = []
    subj_obj = []

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

    return subj, obj, subj_obj


def fill_tables(obj, subj, subj_obj, predicates):
    # Fill tables of obj, subj, and subj_obj where each row has counts of the number of appearances of each predicate
    # for each unique thing
    prep_subj = np.zeros((len(subj), len(predicates)))
    prep_obj = np.zeros((len(obj), len(predicates)))
    prep_subj_obj = np.zeros((len(subj_obj), len(predicates)))

    for pred in predicates:
        data = get_data(pred)
        for i in range(len(data)):
            prep_obj[obj.index(data.iloc[i, 2]), predicates.index(pred)] += 1
            prep_subj[subj.index(data.iloc[i, 9]), predicates.index(pred)] += 1
            prep_subj_obj[subj_obj.index((data.iloc[i, 9], data.iloc[i, 2])), predicates.index(pred)] += 1
        print("fill_tables:", pred, "done")

    return prep_subj, prep_obj, prep_subj_obj


def pickle_file(table, name):
    # https://wiki.python.org/moin/UsingPickle
    file_name = "pred_occurrences/" + name + ".p"
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(table, fp)


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

            # print("label_names[" + str(i) + "]: " + str(label_names[i]))

            # Save image
            if table_name == "subj_obj":
                plt.savefig("pie_chart/" + table_name + "/" + label_names[i][0] + "_" + label_names[i][1] +
                            "_pie_chart.pdf")
            else:
                plt.savefig("pie_chart/" + table_name + "/" + label_names[i] + "_pie_chart.pdf")

            # plt.show()


def get_most_frequent(table_name, most_frequent_table, predicates):
    # If retrieve table from pickled file
    # https://wiki.python.org/moin/UsingPickle
    table_location = "pred_occurrences/prep_" + table_name + ".p"
    table = pickle.load(open(table_location, "rb"))

    for i in range(len(table)):
        for j in range(len(predicates)):
            most_frequent_table[j, i] = table[i, j]


def to_prep_pie_graph(table_name, predicates):
    # If retrieve table from pickled file
    # https://wiki.python.org/moin/UsingPickle
    table_location = "pred_occurrences/most_frequent_" + table_name + ".p"
    table = pickle.load(open(table_location, "rb"))

    # Retrieve labels for each row in table
    label_location = "pred_occurrences/" + table_name + ".p"
    label_names = pickle.load(open(label_location, "rb"))

    for i in range(len(table)):
        most_frequent_num = 20
        most_frequent_indices = table[i].argsort()[-most_frequent_num:][::-1]
        row_sum = int(sum(table[i]))
        legend_labels = []
        for j in range(most_frequent_num):
            legend_label = label_names[most_frequent_indices[j]][0] + ", " + label_names[most_frequent_indices[j]][1] \
                           + ": " + str(int(table[i][most_frequent_indices[j]])) + "/" + str(row_sum) + " (" + \
                           str("{0:.2f}".format(table[i][most_frequent_indices[j]] / row_sum * 100)) + "%)"
            legend_labels.append(legend_label)

        # Create pie chart
        label_name = predicates[i]
        plt.figure(figsize=(14, 7))
        plt.title(label_name)
        patches, texts = plt.pie(table[i][most_frequent_indices], startangle=90, counterclock=False)
        plt.legend(patches, legend_labels, title="Predicates", loc="best")

        # Keep the pie charts circular, not oval
        plt.axis('equal')

        # Save image
        plt.savefig("pred_pie_chart/" + table_name + "/" + predicates[i] + "_pie_chart.pdf")

        # plt.show()


def main():
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # # Get all unique subj, obj, and (subj, obj)
    # subj, obj, subj_obj = get_obj_and_subj(predicates)
    #
    # # Sort the subj, obj, and (subj, obj) lists
    # subj.sort()
    # obj.sort()
    # subj_obj.sort()
    #
    # # Fill the tables (inputting 3 at once to reduce number of calls)
    # prep_subj, prep_obj, prep_subj_obj = fill_tables(obj, subj, subj_obj, predicates)
    #
    # # Save obj, subj, subj_obj as pickle files
    # pickle_file(obj, "obj")
    # pickle_file(subj, "subj")
    # pickle_file(subj_obj, "subj_obj")
    #
    # # Save prep_obj, prep_subj, prep_subj_obj as pickle files
    # pickle_file(prep_obj, "prep_obj")
    # pickle_file(prep_subj, "prep_subj")
    # pickle_file(prep_subj_obj, "prep_subj_obj")

    # # Generate Pie Graphs for each table subj, obj, and (subj, obj) (Retrieve tables from pickled files)
    # to_pie_graph("subj_obj", predicates)
    # to_pie_graph("subj", predicates)
    # to_pie_graph("obj", predicates)

    # # Generate Pie Charts for each predicate and its 10 or 20 or whatever most frequent obj/subj/subj, obj
    # # Want size of subj_obj to do next thing
    # prep_subj_obj_size = len(pickle.load(open("pred_occurrences/prep_subj_obj.p", "rb")))
    #
    # # Initialize most frequent obj/subj/subj_obj of each predicate tables
    # prep_most_frequent_subj_obj = np.zeros((len(predicates), prep_subj_obj_size))
    #
    # # Fill predicate most frequent obj/subj/subj_obj tables
    # get_most_frequent("subj_obj", prep_most_frequent_subj_obj, predicates)
    #
    # # Save most frequent obj/subj/subj_obj tables
    # pickle_file(prep_most_frequent_subj_obj, "most_frequent_subj_obj")

    # Plot predicate most frequent obj/subj/subj_obj tables
    # to_prep_pie_graph("subj_obj", predicates)


    # Get only in/on subj and obj

    predicates = ["in", "on"]

    # Get all unique subj, obj, and (subj, obj)
    subj, obj, subj_obj = get_obj_and_subj(predicates)

    # Sort the subj, obj, and (subj, obj) lists
    subj.sort()
    obj.sort()
    subj_obj.sort()

    # Fill the tables (inputting 3 at once to reduce number of calls)
    prep_subj, prep_obj, prep_subj_obj = fill_tables(obj, subj, subj_obj, predicates)

    # Save obj, subj, subj_obj as pickle files
    pickle_file(obj, "in_on_obj")
    pickle_file(subj, "in_on_subj")
    pickle_file(subj_obj, "in_on_subj_obj")

    # Save prep_obj, prep_subj, prep_subj_obj as pickle files
    pickle_file(prep_obj, "in_on_prep_obj")
    pickle_file(prep_subj, "in_on_prep_subj")
    pickle_file(prep_subj_obj, "in_on_prep_subj_obj")


if __name__ == "__main__":
    main()
