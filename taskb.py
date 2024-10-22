import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img(file):
    img = cv2.imread(file, 0)

    return img, file[2]

def img_threshold(img):
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    img_bin[img_bin == 0] = 1
    img_bin[img_bin == 255] = 0

    return img_bin

def hist_projection(img_bin):
    v_proj = np.sum(img_bin, axis=0)
    h_proj = np.sum(img_bin, axis=1)

    return v_proj, h_proj 

def img_dilate(img_bin):
    kernel = np.ones((5, 5), np.uint8) 
    res = cv2.dilate(img_bin, kernel, iterations= 1)
    return res

# Obtaining Col Threshold from Image
def get_col_threshold(img_bin):
    v_proj = np.sum(img_bin, axis=0)
    iszero = np.equal(v_proj, 0).view(np.int8)
    absdiff = np.abs(np.diff(iszero))
    col_ranges = np.where(absdiff == 1)[0]
    col_start, col_end = col_ranges[0], col_ranges[-1]
    col_threshold = [(col_ranges[i], col_ranges[i+1]) for i in range(1,len(col_ranges) - 1,2)]
    col_threshold = [a for a in col_threshold if a[1] - a[0] > 250]
    col_threshold = [(0, col_start)] + col_threshold + [(col_end, len(v_proj)- 1)]

    return col_threshold


# Obtaining Row Threshold from Image
def get_row_threshold(a,b, img_bin):
    h_proj = np.sum(img_bin[:, a:b], axis=1)
    iszero = np.equal(h_proj, 0).view(np.int8)
    absdiff = np.abs(np.diff(iszero))
    row_ranges = np.where(absdiff == 1)[0]
    row_start, row_end = row_ranges[0], row_ranges[-1]
    row_threshold = [(row_ranges[i],row_ranges[i+1]) for i in range(1,len(row_ranges)-1 ,2)]
    row_threshold = [a for a in row_threshold if a[1] - a[0] > 250]

    row_threshold = [(0, row_start)] + row_threshold + [(row_end, len(h_proj)- 1)]

    return row_threshold, h_proj


def process(img, img_bin, num):
    col_threshold = get_col_threshold(img_bin)
    col_width = col_threshold[-1][0] - col_threshold[0][1]
    dilated = dilate(img_bin)
    v, h = hist_projection(img_bin)

    for i in range(len(col_threshold) - 1):
        start_y_mid = (col_threshold[i][0] + col_threshold[i][1]) // 2
        end_y_mid = (col_threshold[i+1][0] + col_threshold[i+1][1]) // 2

        row_threshold, h_proj = get_row_threshold(start_y_mid, end_y_mid, img_bin)

        for j in range(len(row_threshold) - 1):
            if h_proj[row_threshold[j][1] + 1:row_threshold[j+1][0] + 1].min() == 0:
                start_x_mid = (row_threshold[j][0] + row_threshold[j][1]) // 2
                end_x_mid = (row_threshold[j+1][0] + row_threshold[j+1][1]) // 2

                extracted = img[start_x_mid:end_x_mid, start_y_mid:end_y_mid]

                if(h.max() / col_width >= 0.3):
                    extracted_bin = img_threshold(extracted)
                    process(extracted, extracted_bin, num)
                else:
                    cv2.imwrite(f"./taskb/image-{num}/col-{i + 1}-paragraph-{j + 1}.png", extracted)
                    print(num)
            else:
                continue


def main():
    images = ["001.png", "002.png", "003.png", "004.png", "005.png", "006.png", "007.png", "008.png"]
    for i in images:
        img, num = read_img(i)
        
        img_bin = img_threshold(img)
        process(img, img_bin, num)


if __name__=="__main__": 
    main() 

