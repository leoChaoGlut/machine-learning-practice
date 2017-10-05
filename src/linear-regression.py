import pymysql.cursors
import sys
from sklearn import linear_model
# from pylab import plot, show
# from numpy import asarray
# from datetime import datetime
from decimal import *


# import time


def loadData(storeId, productId):
    connection = pymysql.connect(
        host='localhost',
        user='admin',
        password='admin',
        db='test',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        feature = []
        label = []
        with connection.cursor() as cursor:
            sql = """
                SELECT
                  old_inventory_count,
                  unix_timestamp(create_time) ct
                FROM test.inventory_change ic
                  JOIN test.inventory i ON ic.inventory_id = i.id
                WHERE i.store_id = %s AND i.product_id = %s
                ORDER BY create_time
            """
            cursor.execute(sql, (storeId, productId))
            result = cursor.fetchall()
            for res in result:
                inventoryCount = res['old_inventory_count']
                ct = res['ct']
                # print(ct, inventoryCount)
                feature.append([float(ct)])
                label.append(float(inventoryCount))
            return feature, label
    finally:
        connection.close()


def calc(storeId, productId):
    feature, label = loadData(sys.argv[1], sys.argv[2])

    if len(feature) == 0 or len(label) == 0:
        print(0, 0, end="")
        exit(0)

    reg = linear_model.Lasso()
    reg.fit(feature, label)
    # x = asarray(feature)[:, 0]
    # func = reg.coef_ * x + reg.intercept_
    # func1 = -1.81213665 * x + 25936.590418600725
    # print(func)
    # print(func1)
    print(Decimal.from_float(reg.coef_[0]), reg.intercept_, end="")


# print(-reg.intercept_ / reg.coef_)
# print(reg.predict([[1484749220]]))
# print(datetime.fromtimestamp(-reg.intercept_ / reg.coef_[0]))
# plot(x, label, 'o')
# plot(x, func, 'k-')
# plot(x, func1, 'k')
# show()

"""
y=ax+b
1484649220
1485804690
when y==0:
    0=ax+b
    ax=-b
    x=-b/a    
"""
