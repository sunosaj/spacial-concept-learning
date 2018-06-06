from PyPDF2 import PdfFileMerger, PdfFileReader
from fpdf import FPDF
from os import listdir
from os.path import isfile, join


def main():
    merger = PdfFileMerger()

    # Can only merge one set right now, so whatever it is to be combined, comment out everything else

    # # Merge plots
    # predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
    #               "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
    #               "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
    #               "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
    #               "within", "without"]
    #
    # for pred in predicates:
    #     file_address = 'new_pred_plot/' + pred + '_plot.png'
    #     merger.append(PdfFileReader(file_address, 'rb'))
    #
    # merger.write("new_pred_plot/combined_plot.png")

    # # Merge pngs to one pdf
    # predicates = ["about", "above", "across", "after", "against", "along", "alongside", "amid", "amidst", "around",
    #               "at", "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "from", "in",
    #               "inside", "into", "near", "off", "on", "onto", "opposite", "out", "outside", "over", "past",
    #               "stop", "through", "throughout", "to", "toward", "under", "underneath", "up", "upon", "with",
    #               "within", "without"]
    # pdf = FPDF()
    # # imagelist is the list with all image filenames
    # for pred in predicates:
    #     file_address = 'new_pred_plot/' + pred + '_plot.png'
    #     pdf.add_page()
    #     pdf.image(file_address)
    # pdf.output("combined_plot.pdf", "F")

    # # Merge pie charts
    # category_names = ['obj', 'subj', 'subj_obj']
    # for category in category_names:
    #     print('category:', category)
    #     mypath = 'pie_chart/' + category + '/'
    #     print('mypath:', mypath)
    #     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #     if 'combined_' + category + '_pie_chart.pdf' in onlyfiles:
    #         onlyfiles.remove('combined_' + category + '_pie_chart.pdf')
    #     print('onlyfiles:', onlyfiles)
    #     for file in onlyfiles:
    #         merger.append(PdfFileReader(mypath + file, 'rb'))
    #
    #     merger.write(mypath + 'combined_' + category + '_pie_chart.pdf')

    # # Merge obj pie charts
    # print('category: obj')
    # mypath = 'pie_chart/obj/'
    # print('mypath:', mypath)
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # if 'combined_obj_pie_chart.pdf' in onlyfiles:
    #     onlyfiles.remove('combined_obj_pie_chart.pdf')
    # print('onlyfiles:', onlyfiles)
    # for file in onlyfiles:
    #     merger.append(PdfFileReader(mypath + file, 'rb'))
    #
    # merger.write(mypath + 'combined_obj_pie_chart.pdf')

    # # Merge subj pie charts
    # print('category: subj')
    # mypath = 'pie_chart/subj/'
    # print('mypath:', mypath)
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # if 'combined_subj_pie_chart.pdf' in onlyfiles:
    #     onlyfiles.remove('combined_subj_pie_chart.pdf')
    # print('onlyfiles:', onlyfiles)
    # for file in onlyfiles:
    #     merger.append(PdfFileReader(mypath + file, 'rb'))
    #
    # merger.write(mypath + 'combined_subj_pie_chart.pdf')

    # # Merge (subj, obj) pie charts
    # print('category: subj_obj')
    # mypath = 'pie_chart/subj_obj/'
    # print('mypath:', mypath)
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # if 'combined_subj_obj_pie_chart.pdf' in onlyfiles:
    #     onlyfiles.remove('combined_subj_obj_pie_chart.pdf')
    # print('onlyfiles:', onlyfiles)
    # for file in onlyfiles:
    #     merger.append(PdfFileReader(mypath + file, 'rb'))
    #
    # merger.write(mypath + 'combined_subj_obj_pie_chart.pdf')

    # Merge pred subj, obj pie charts
    print('category: pred, subj_obj')
    mypath = 'pred_pie_chart/subj_obj/'
    print('mypath:', mypath)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    if 'combined_subj_obj_pie_chart.pdf' in onlyfiles:
        onlyfiles.remove('combined_pred_subj_obj_pie_chart.pdf')
    print('onlyfiles:', onlyfiles)
    for file in onlyfiles:
        merger.append(PdfFileReader(mypath + file, 'rb'))

    merger.write(mypath + 'combined_pred_subj_obj_pie_chart.pdf')


if __name__ == "__main__":
    main()