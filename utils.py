import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_info(target, level):
    
    if target == 'puzzle':
        pages = os.listdir('data/pages')
        WIDTH, HEIGHT = 500, 470
        file_name = 'data/pages/{}_page_{}.jpg'
        is_solution = False

    else:
        pages = os.listdir('data/solutions')
        file_name = 'data/solutions/solution_{}_{}.jpg'
        WIDTH, HEIGHT = 350, 330
        is_solution = True
        
    pages = [p for p in pages if level in p]
    
    return pages, file_name, WIDTH, HEIGHT, is_solution


def divide_page(page, WIDTH, HEIGHT):
    
    puzzles = []
    
    for row in range(3):
        for col in range(2):
            
            puzzle = page[HEIGHT*row:HEIGHT*(row+1), WIDTH*col:WIDTH*(col+1)]
            
            puzzles.append(puzzle)
            
    return puzzles


def get_bboxes(puzzles):
    
    boxes = []
    
    for puzzle in puzzles:
        
        # The outer line was thin that without erosion, it could not detect it well
        # Make all lines thicker a bit
        puzzle = cv2.erode(puzzle, np.ones((3,3))).astype('uint8')

        contours, _ = cv2.findContours(puzzle, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        box = []

        for c in contours:

            rect = cv2.boundingRect(c)
            area = rect[2] * rect[3]

            box.append((rect, area))

        # Sort by area in decreasing order
        box.sort(key=lambda x:x[1], reverse=True)

        # Skip the first one as it covers the entire image
        box = box[1]
        
        boxes.append(box)
        
    return boxes


def extract_puzzle(puzzles, bboxes):
    
    extracted = []
    
    for puzzle, bbox in zip(puzzles, bboxes):

        # The second item is the area
        x, y, w, h = bbox[0]
        
        puzzle = puzzle[y:y+h, x:x+w]
        
        extracted.append(puzzle)
        
    return extracted



def plot_image(img, cmap=None, figsize=(8,8)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    

def get_puzzle(img):
    
    # The outer line was thin that without erosion, it could not detect it well
    # Make all lines thicker a bit

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.erode(gray, np.ones((3,3)))

    contours = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]

    a = gray.copy()
    box = []

    for c in contours:

        rect = cv2.boundingRect(c)
        area = rect[2] * rect[3]

        box.append((rect, area))

    # Sort by area in decreasing order
    box.sort(key=lambda x:x[1], reverse=True)

    # Skip the first one as it covers the entire image
    box = box[1:]
    x, y, w, h = box[0][0]
    # Find a box whose y is less than box[0]'s y
    # Note there could be multiple boxes for label
    # Extract the box with largest area
    largest_area = 0
    label = []
    
    for b in box[1:]:

        b_x, b_y, b_w, b_h = b[0]
        area = b_w * b_h

        # If above puzzle
        if b_y < y:
            label.append(b[0])

    # There could be two digits for label
    # In this case calculate the bounding box for two or more
    if len(label) > 1:
        label = np.array(label)
        
    # Don't return the last elem in box[0] since it is the area
    return box[0][0], label


def draw_box(img, box, label):
    
    box_used = []
    temp_img = img.copy()

    x, y, w, h = box
    b_x, b_y, b_w, b_h = label

    # Add padding around bounding box
    # Make sure values are within valid range
    x = max(0, x)
    y = max(0, y)
    w = min(a.shape[0], w)
    h = min(a.shape[1], h)

    temp_img = cv2.rectangle(temp_img, (x, y), (x+w, y+h), 255, 2)
    temp_img = cv2.rectangle(temp_img, (b_x, b_y), (b_x+b_w, b_y+b_h), 255, 2)
    temp_img = cv2.addWeighted(temp_img, .9, p1, 1-.9, 0)
    
    return temp_img


def extract_digit(img, cell_w, cell_h, model):

    answers = []

#     fig, ax = plt.subplots(9, 9, figsize=(12, 12))
    
    for i in range(9):
        for j in range(9):

            start_x = cell_w*j
            start_y = cell_h*i

            end_x = cell_w*(j+1)
            end_y = cell_h*(i+1)

            cell = img[start_y:end_y, start_x:end_x]
            
#             ax[i][j].imshow(cell)

            cell = cell.reshape((1, cell_w, cell_h, 1))

            pred = model.predict(cell)
            pred = pred.argmax(-1)[0]

            answers.append(pred)

    answers = np.array(answers)
    
    return answers


def save_puzzle(img, label, puzzle_base, level, is_solution):
    
    if is_solution:
        name = 'data/puzzles/{}_{}_solution'.format(level, label)
    else:
        name = 'data/puzzles/{}_{}'.format(level, label)
    
    plt.imsave(name+'.jpg', img)

    with open('{}'.format(name+'.txt'), 'w') as file:
        
        file.writelines(puzzle_base)
        
        
        
def get_digits(img, model, sup=False, dense=True):
    
    digits = []
    
    for row in range(9):
        for col in range(9):
            
            cell = img[row*28:(row+1)*28, col*28:(col+1)*28]

            if sup:
                pred = model.predict(cell.reshape(1, -1))
                pred = int(pred[0])
                
            elif dense:
                pred = model.predict(cell.reshape(1, -1))
                pred = pred.argmax(-1)[0]
                
            else:
                pred = model.predict(cell.reshape(1, 28, 28, 1))
                pred = pred.argmax(-1)[0]

            digits.append(pred)
            
    return np.array(digits).reshape((9, 9))



def clean_puzzle(puzzle, is_solution):
    
    w, h = puzzle.shape[1]//9, puzzle.shape[0]//9
    
    for row in range(9):
        for col in range(9):
            cell = puzzle[row*h:(row+1)*h, col*w:(col+1)*w]
            cell[:5, :] = 255
            cell[-5:, :] = 255
            cell[:, :5] = 255
            cell[:, -5:] = 255

            if not is_solution:
                cell[:8, :] = 255
                cell[:, :8] = 255
                cell[:, -6:] = 255

            puzzle[row*h:(row+1)*h, col*w:(col+1)*w] = cell
    return puzzle