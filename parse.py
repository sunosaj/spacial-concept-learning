import json
import csv

image_dimensions = {}


def get_image_dimensions():
    read_file = json.loads(open('image_data.json', 'r').read())

    for image in read_file:
        image_dimensions[image["image_id"]] = [image["width"], image["height"]]


def json_to_csv(pred):
    read_file = json.loads(open('relationships.json', 'r').read())
    new_file_name = "new_pred/data_pred_" + pred.lower() + ".csv"
    write_file = csv.writer(open(new_file_name, "wb+"))
    write_file.writerow(["image_id", "predicate", "obj_name", "obj_h", "obj_w", "obj_x", "obj_y", "obj_id", "id",
                         "subj_name", "subj_h", "subj_w", "subj_x", "subj_y", "subj_id", "image_width", "image_height"])

    for rela in read_file:
        image_id = rela["image_id"]
        image_width = image_dimensions[image_id][0]
        image_height = image_dimensions[image_id][1]
        for row in rela["relationships"]:
            obj_name = row["object"]["name"] if "name" in row["object"] else row["object"]["names"][0]
            subj_name = row["subject"]["name"] if "name" in row["subject"] else row["subject"]["names"][0]

            if pred.lower() == row["predicate"].lower():
                write_file.writerow([image_id,
                                    row["predicate"],
                                    obj_name,
                                    row["object"]["h"],
                                    row["object"]["w"],
                                    row["object"]["x"],
                                    row["object"]["y"],
                                    row["object"]["object_id"],
                                    row["relationship_id"],
                                    subj_name,
                                    row["subject"]["h"],
                                    row["subject"]["w"],
                                    row["subject"]["x"],
                                    row["subject"]["y"],
                                    row["subject"]["object_id"],
                                    image_width,
                                    image_height])


def main():
    get_image_dimensions()
    predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "nearby", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    for predicate in predicates:
        json_to_csv(predicate)


if __name__ == "__main__":
    main()
