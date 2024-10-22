import numpy as np
import cv2

# Function to read the image file
def readImg(img_file):
    img = cv2.imread(img_file, 0)
    
    return img, img_file[2] # Return the image array and the numbering of the image file

# Function to perform pre-processing for paragraph extraction
def imgPreprocess(img):
    bin_img = cv2.threshold(img, 30, 1, cv2.THRESH_BINARY_INV)[1] # Obtain the binary image
    
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) # Generate structuring element
    
    dilat_img = cv2.dilate(bin_img, SE, iterations = 4) # Dilate the binary image
    
    return bin_img, dilat_img # Return the binary image and the dilated binary image

# Function to perform the actual paragraph extraction process
def extProj(img_num, img, dilat_img):
    # Lists to store starting and ending of row and column indices of the extracted image section
    row_starts, row_ends, col_starts, col_ends = [], [], [], []
    # A list to track which column a row belongs to
    col_of_row = []
    
    col_proj = np.sum(dilat_img, axis = 0) # Calculate column projection
    
    '''
    Logic to determine starting and ending pairs within the projections
    
    Start index - the index of an element with a non-zero value preceded by an element with a value of 0
    End index - the index of an element with a non-zero value followed by an element with a value of 0
    
    [Example]
    
    0 - zero value
    x - non-zero value
    
    0000000000000xxxxxxxxxxxxxxxx00000000000000000 (in the projections)
                 ^              ^
                start          end
    '''
    
    # Determine start and end pairs for column
    for i in range(0, len(col_proj) - 1):
        if col_proj[i] == 0 and col_proj[i + 1] != 0:
            col_starts.append(i + 1)
        elif col_proj[i] != 0 and col_proj[i + 1] == 0:
            col_ends.append(i)
    
    # Determine start and end pairs for row
    # The row projections calculated must be constrained by the columns in the image
    # That way, the correct row indices can be obtained according to the column they belong to
    
    # Iterate through the column starting & ending index
    for i in range(0, len(col_starts)):
        # Calculate the row projection based on col_start and col_end for each iteration
        row_proj = np.sum(dilat_img[:, col_starts[i]:col_ends[i] + 1], axis = 1)
        
        # Determine start and end pairs for row
        for j in range(0, len(row_proj) - 1):
            if row_proj[j] == 0 and row_proj[j + 1] != 0:
                row_starts.append(j + 1)
                col_of_row.append(i)
            elif row_proj[j] != 0 and row_proj[j + 1] == 0:
                row_ends.append(j)
    
    para_num = 1 # Variable to keep track of the paragraph number
    
    # A for-loop to extract the paragraph from the image
    for col in range(0, len(col_starts)):
        for row in range(0, len(row_starts)):
            if col_of_row[row] == col: # Check if the row indices correspond to its column in the current iteration
            
                # Extract the image section that might contain a paragraph
                # Some padding is added to all 4 directions of the extracted image section
                ext_img = img[row_starts[row] - 10:row_ends[row] + 11, col_starts[col] - 10:col_ends[col] + 11]
                
                # Obtain the binary image & dilated binary image of the extracted section for validation
                bin_ext_img, dilat_ext_img = imgPreprocess(ext_img)
                
                # Check if the extracted portion is a paragraph
                if validatePara(bin_ext_img):
                    
                    # Check if the extracted section consists of multiple columns (can be extracted further)
                    if checkSubColumn(dilat_ext_img):
                        
                        # If the extracted section has subcolumns
                        # Repeat the operation by passing in the extracted section along with the parameters
                        # So that a non-paragraph element that spans across the image (such as the table in 004.png) will not disrupt the process
                        # And each individual paragraphs can be extracted correctly
                        extProj(img_num, ext_img, dilat_ext_img)
                    else:
                        # Write the extracted section into an image file if it passes every validation
                        cv2.imwrite(f"./taskb/image-{img_num}/paragraph-00{para_num}.png", ext_img)
                        para_num += 1 # increment the paragraph number for each successful image file written

# Function to validate if an extracted portion is a paragraph
def validatePara(bin_ext_img):
    
    # Obtain the row projection of the binarised extracted section
    ext_row_proj = np.sum(bin_ext_img, axis = 1)
    
    # From the top of the row projection, find the first starting row index
    for i in range(0, len(ext_row_proj) - 1):
        if ext_row_proj[i] == 0 and ext_row_proj[i + 1] != 0:
            start = i + 1
            break
    
    # From the bottom of the row projection, find the last ending row index
    for i in range(-1, -len(ext_row_proj) + 1, -1):
        if ext_row_proj[i] == 0 and ext_row_proj[i - 1] != 0:
            end = i - 1
            break
    
    # Use the previously obtained indices to get the portion of the extracted section that only contains the content
    # Basically it's to obtain a padding-free version of the extracted section (using the start & end index)
    content = ext_row_proj[start:end - 1]
    
    '''
    The extracted section may contain paragraph, image or table.
    Each of these content have the following characteristics:
        Image -> No element with a value of 0 in row projection
        Table -> Has element with a value of 0 (minimum value = 0) in the row projection, but the number of occurrences is <= 3
        Paragraph -> Has element with a value of 0 (minimum value = 0) in the row projection, with the number of occurrences >= 10
    Using these characteristics, the contents of the extracted portion can be identified
    
    Using a elimination process, 2 conditions are checked
    First condition:
        Determine the minimum value of the projection of that section
        If the minimum value of the projection is 0 -> extracted section is not an image
    
    Second condition:
        Count the occurrences of the element with a value of 0 in the row projection
        If the occurrences of 0s in the row projection is >= 10 -> extracted section is not a table
        
        NOTE: 
            - Initially we thought the first condition is sufficient to filter out images and tables
            - But we only managed to filter the images, but not tables
            - After a manual inspection of the row projection, we found out that sections that contain tables can have zero value
            - The number of 0s is not more than 3 for tables
            - So we introduced this 2nd condition
    
    If the extracted section passes these two conditions, it must be a paragraph
    '''
    
    if (min(content) == 0 and np.count_nonzero(content == 0) >= 10):
        return True # If the extracted portion only consists of texts (paragraph), return True
    return False # Return false if otherwise

# Function to check if the extracted paragraph consists of multiple columns
def checkSubColumn(dilat_ext_img):
    
    # Get the projection of the dilated binary image of the extracted portion
    ext_col_proj = np.sum(dilat_ext_img, axis = 0)
    
    col_starts= []
    
    ''' This is where the padding for the extracted portion comes in handy'''
    # We only check for either start index or end index to calculate the number of columns in the extracted section
    # With padding, there will still be empty spaces in all 4 directions after in the dialted bianry image
    # This allows for locating the start/end column indices using the same method
    for i in range(0, len(ext_col_proj) - 1):
        if ext_col_proj[i] == 0 and ext_col_proj[i + 1] != 0:
            col_starts.append(i + 1)
        
    # If the extracted portion consists of more than 1 column, return True and False if otherwise
    if len(col_starts) > 1:
        return True
    return False
    

# Main method/function for this program
def main():
    # List containing the filename of the images (strings)
    images = ["001.png", "002.png", "003.png", "004.png", "005.png", "006.png", "007.png", "008.png"]
    
    # The loop to start the paragraph extraction process for each image
    for img in images:
        image, image_num = readImg(img)
        bin_image, dilat_image = imgPreprocess(image)
        extProj(image_num, image, dilat_image)
    
    
if __name__ == "__main__":
    main()