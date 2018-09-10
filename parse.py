import json
import csv


def get_image_dimensions():
    image_dimensions = {}

    read_file = json.loads(open('image_data.json', 'r').read())

    for image in read_file:
        image_dimensions[image["image_id"]] = [image["width"], image["height"]]

    return image_dimensions


def json_to_csv(prep, image_dimensions):
    read_file = json.loads(open('relationships.json', 'r').read())
    new_file_name = "new_prep/data_prep_" + prep.lower() + ".csv"
    write_file = csv.writer(open(new_file_name, "w+", newline=''))
    write_file.writerow(["image_id", "preposition", "obj_name", "obj_h", "obj_w", "obj_x", "obj_y", "obj_id", "id",
                         "subj_name", "subj_h", "subj_w", "subj_x", "subj_y", "subj_id", "image_width", "image_height"])

    for relationship in read_file:
        image_id = relationship["image_id"]
        image_width = image_dimensions[image_id][0]
        image_height = image_dimensions[image_id][1]
        for row in relationship["relationships"]:
            obj_name = row["object"]["name"] if "name" in row["object"] else row["object"]["names"][0]
            subj_name = row["subject"]["name"] if "name" in row["subject"] else row["subject"]["names"][0]

            if prep.lower() == row["preposition"].lower():
                write_file.writerow([image_id,
                                    row["preposition"],
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
    prepositions = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
                  "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
                  "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
                  "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
                  "within", "without"]

    # prepositions = ["above"]

    image_dimensions = get_image_dimensions()

    for preposition in prepositions:
        json_to_csv(preposition, image_dimensions)


if __name__ == "__main__":
    main()
