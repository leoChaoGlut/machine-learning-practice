import numpy as np
import pandas as pd

df = pd.DataFrame(
    data=np.zeros((3, 3)),
    index=['x1', 'x2', 'x3'],
    columns=['y1', 'y2', 'y3']
)

print(df)


for x in range(4):
    print(x)