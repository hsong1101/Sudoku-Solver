```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from utils import *
```


```python
with open('data/digits/digits_puzzle.txt', 'r') as file:
    digits_puzzle = file.readlines()
    
with open('data/digits/digits_solution.txt', 'r') as file:
    digits_solution = file.readlines()
    
digits_puzzle = np.array([x.replace('\n', '').split(' ') for x in digits_puzzle], dtype=int)
digits_solution = np.array([x.replace('\n', '').split(' ') for x in digits_solution], dtype=int)
```


```python
digits_puzzle.shape, digits_solution.shape
```




    ((6, 81), (6, 81))




```python
img = pd.read_csv('data/puzzles/easy_1.csv').values

plt.imshow(img)
digits_puzzle[0].reshape((9,9))
```




    array([[0, 0, 0, 0, 8, 0, 0, 0, 9],
           [0, 0, 0, 4, 2, 6, 1, 3, 0],
           [0, 0, 0, 9, 0, 1, 5, 0, 6],
           [2, 0, 0, 8, 3, 0, 9, 7, 4],
           [3, 0, 9, 0, 6, 0, 0, 8, 0],
           [0, 0, 0, 2, 9, 4, 0, 0, 0],
           [0, 5, 6, 3, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 8, 0, 7],
           [0, 8, 4, 0, 5, 2, 0, 1, 0]])




![png](extract_digits_files/extract_digits_3_1.png)



```python
img = pd.read_csv('data/puzzles/easy_1_solution.csv').values

plt.imshow(img)
digits_solution[0].reshape((9,9))
```




    array([[6, 1, 2, 5, 8, 3, 7, 4, 9],
           [5, 9, 7, 4, 2, 6, 1, 3, 8],
           [4, 3, 8, 9, 7, 1, 5, 2, 6],
           [2, 6, 1, 8, 3, 5, 9, 7, 4],
           [3, 4, 9, 1, 6, 7, 2, 8, 5],
           [8, 7, 5, 2, 9, 4, 3, 6, 1],
           [7, 5, 6, 3, 1, 8, 4, 9, 2],
           [1, 2, 3, 6, 4, 9, 8, 5, 7],
           [9, 8, 4, 7, 5, 2, 6, 1, 3]])




![png](extract_digits_files/extract_digits_4_1.png)



```python
data = []

for i in range(1, 7):

    puzzle = pd.read_csv('data/puzzles/easy_{}.csv'.format(i)).values
    solution = pd.read_csv('data/puzzles/easy_{}_solution.csv'.format(i)).values

    w, h = puzzle.shape[1]//9, puzzle.shape[0]//9
    puz_txt = digits_puzzle[i-1].reshape((9, 9))
    sol_txt = digits_solution[i-1].reshape((9, 9))

    for row in range(9):
        for col in range(9):

            puz_digit = puz_txt[row][col]
            sol_digit = sol_txt[row][col]

            puz_cell = puzzle[row*h:(row+1)*h, col*w:(col+1)*w].flatten().tolist()
            sol_cell = solution[row*h:(row+1)*h, col*w:(col+1)*w].flatten().tolist()

            data.append([puz_digit] + puz_cell)
            data.append([sol_digit] + sol_cell)

data = np.array(data)
```


```python
data.shape
```




    (972, 1765)




```python
num_cells = 16
fig, ax = plt.subplots(num_cells, num_cells, figsize=(9, 9))

labels = []

for i in range(num_cells):
    for j in range(num_cells):
        
        idx = np.random.randint(data.shape[0])
        labels.append(data[idx][0])
        x = data[idx][1:]
        x = x.reshape((w, h))
        
        ax[i][j].imshow(x)
        
np.array(labels).reshape((num_cells, num_cells))
```




    array([[0, 9, 6, 0, 1, 3, 3, 0, 5, 7, 2, 9, 0, 3, 2, 3],
           [9, 8, 7, 0, 0, 1, 7, 2, 4, 6, 0, 5, 8, 6, 6, 0],
           [5, 4, 0, 6, 7, 0, 7, 9, 0, 5, 7, 8, 1, 2, 8, 4],
           [3, 3, 0, 5, 2, 0, 9, 0, 0, 9, 8, 7, 9, 0, 0, 4],
           [5, 0, 1, 5, 0, 0, 9, 9, 7, 2, 8, 5, 9, 3, 3, 2],
           [0, 0, 8, 7, 0, 2, 1, 5, 0, 8, 0, 9, 0, 0, 4, 4],
           [0, 2, 4, 3, 0, 7, 4, 2, 4, 5, 2, 0, 1, 5, 0, 0],
           [0, 0, 0, 4, 6, 7, 4, 1, 1, 0, 2, 5, 2, 9, 0, 7],
           [1, 4, 0, 2, 3, 5, 0, 7, 6, 0, 3, 0, 4, 9, 7, 5],
           [0, 0, 0, 3, 9, 9, 2, 9, 7, 4, 0, 4, 5, 0, 0, 4],
           [0, 1, 1, 0, 3, 3, 9, 6, 0, 0, 7, 5, 0, 0, 6, 0],
           [2, 7, 8, 5, 8, 6, 9, 0, 5, 4, 1, 2, 7, 5, 8, 0],
           [7, 2, 0, 5, 4, 8, 8, 3, 0, 9, 8, 0, 1, 0, 5, 9],
           [5, 9, 9, 4, 0, 5, 3, 0, 4, 0, 0, 9, 0, 3, 0, 8],
           [9, 1, 2, 0, 0, 0, 0, 4, 4, 6, 0, 8, 8, 0, 6, 6],
           [1, 5, 6, 1, 9, 0, 0, 0, 0, 2, 0, 2, 0, 4, 0, 8]])




![png](extract_digits_files/extract_digits_7_1.png)



```python
pd.DataFrame(data).to_csv('data/digits/data.csv', index=False)
```
