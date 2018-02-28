import pandas as pd

df = pd.DataFrame(
    # data=np.zeros((3, 3)),
    data=[
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ],
    index=['x1', 'x2', 'x3'],
    columns=['y1', 'y2', 'y3']
)

print(df)

print(df.loc['x2', :].max())


# for x in range(40):
#     os.system('clear')
#     print(df)
#     time.sleep(0.5)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equal(self, point):
        return self.x == point.x and self.y == point.y


p1 = Point(1, 3)
p2 = Point(1, 2)
print(p1.equal(p2))
