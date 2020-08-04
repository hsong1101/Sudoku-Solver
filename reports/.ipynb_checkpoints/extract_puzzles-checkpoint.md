```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from utils import *
```


```python
levels = ['easy', 'medium', 'hard', 'extreme']

targets = ['puzzle', 'solution']

for target in targets:
    
    for level in levels:
        
        pages, file_name, WIDTH, HEIGHT, is_solution = get_info(target, level)
        
        quiz_num = 1
        
        print(target, level)
        
        for i, page in enumerate(pages):
            
            file = file_name.format(level, i+1)

            page = plt.imread(file)
            
            gray_page = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
            
            if is_solution:
                
                left_page = gray_page[:, :gray_page.shape[1]//2]
                right_page = gray_page[:, gray_page.shape[1]//2:]
                
                puzzles = divide_page(left_page, WIDTH, HEIGHT)
                
                # Last page of extreme has only left page
                if not level == 'extreme' or not i == 2:
                    puzzles.extend(divide_page(right_page, WIDTH, HEIGHT))
                
            else:
                
                puzzles = divide_page(gray_page, WIDTH, HEIGHT)
            
            bboxes = get_bboxes(puzzles)
            puzzles = extract_puzzle(puzzles, bboxes)

            for puzzle in puzzles:
                
                if is_solution:
                    save_name = 'data/puzzles/{}_{}_solution.csv'.format(level, quiz_num)
                else:
                    save_name = 'data/puzzles/{}_{}.csv'.format(level, quiz_num)

                puzzle = cv2.resize(puzzle, (28*9, 28*9))
                puzzle = puzzle / 255
                
                # Because matplotlib.pyplot doesn't save image in grayscale, I chose pandas to save data
                pd.DataFrame(puzzle).to_csv(save_name, index=False)
                
                quiz_num += 1
                
```

    puzzle easy
    puzzle medium
    puzzle hard
    puzzle extreme
    solution easy
    solution medium
    solution hard
    solution extreme



```python

```
