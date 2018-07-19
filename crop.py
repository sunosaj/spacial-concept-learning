from PyPDF2 import PdfFileWriter, PdfFileReader
# import pyPdf
import cv2
from PIL import Image


def crop_bounding_boxes(file_address):
    with open("new_pred_bb_plot/" + file_address, "rb") as in_f:
        input1 = PdfFileReader(in_f)
        output = PdfFileWriter()

        numPages = input1.getNumPages()

        for i in range(numPages):
            page = input1.getPage(i)
            # page.trimBox.lowerLeft = (10, 10)
            # page.trimBox.upperRight = (225, 225)
            page.cropBox.lowerLeft = (46, 25)
            page.cropBox.upperRight = (310, 289)
            output.addPage(page)

        with open("new_pred_bb_plot/" + file_address[:-4] + "_cropped_pdf.pdf", "wb") as out_f:
            output.write(out_f)


def crop_png(file_address):
    img = cv2.imread("new_pred_bb_plot/" + file_address)
    # print(img.shape)

    # crop_img = img[95:95+2210, 378:378+2210]
    # print(crop_img.shape)
    # cv2.imwrite("new_pred_bb_plot/" + file_address[:-4] + "_cropped.png", crop_img)

    crop_img = img[10:10+202, 35:35+202]
    print("crop_img.shape", crop_img.shape)
    cv2.imwrite("new_pred_bb_plot/" + file_address[:-4] + "_cropped.png", crop_img)


def change_res(file_address):
    # print(file_address)
    im = Image.open("new_pred_bb_plot/" + file_address)

    # im.save("new_pred_bb_plot/" + file_address[:-4] + "_low_res.png", dpi=(100, 100))


def check_num_pixels(file_address):
    print(cv2.imread("new_pred_bb_plot/" + file_address).shape)


def get_image_pixels(file_address):
    im = Image.open("new_pred_bb_plot/" + file_address)
    pix = im.load()
    # print(im.size)

    count_red = 0
    count_close_red = 0
    count_blue = 0
    count_close_blue = 0
    count_white = 0
    count_else = 0

    img_pixels = []
    for i in range(im.size[0]):
        img_pixels_row = []
        for j in range(im.size[1]):
            if pix[i, j] != (255, 255, 255):
                img_pixels_row.append(pix[i, j])
            # red
            if pix[i, j] == (255, 59, 59):
                count_red += 1
            # blue
            elif pix[i, j] == (59, 59, 255):
                count_blue += 1
            # close red
            elif pix[i, j] == (255, 77, 77):
                count_close_red += 1
            # close blue
            elif pix[i, j] == (77, 77, 255):
                count_close_blue += 1
            # white
            elif pix[i, j] == (255, 255, 255):
                count_white += 1
            else:
                count_else += 1
        img_pixels.append(img_pixels_row)
    print(img_pixels)
    print(count_red)
    print(count_close_red)
    print(count_blue)
    print(count_close_blue)
    print(count_white)
    print(count_else)


def main():
    # for i in range(0, 10):
        # crop_bounding_boxes("on_bounding_box_plot_" + str(i) + ".pdf")
        # crop_png("on_bounding_box_plot_" + str(i) + ".png")
        # change_res("on_bounding_box_plot_" + str(i) + "_cropped.png")
        # check_num_pixels("on_bounding_box_plot_" + str(i) + "_cropped_pdf_cropped.png")

        # check_num_pixels("on_bounding_box_plot_" + str(i) + ".png")

    get_image_pixels("on_bounding_box_plot_" + str(4) + "_cropped.png")


if __name__ == "__main__":
    main()
