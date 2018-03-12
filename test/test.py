import pandas as pd

cols = ['up', 'down', 'left', 'right']

df = pd.DataFrame(columns=cols)

# df = df.append(
#     pd.Series(
#         data=[0, 1, 2, 3],
#         index=cols,
#         name='a'
#     )
# )

df = df.append(
    pd.DataFrame(
        data=[[0, 1, 2, 3]],
        columns=cols,
        index=['a']
    )
)

print(df)

if 'b' in df.index:
    print(1)
else:
    print(2)

print([0]*3)
