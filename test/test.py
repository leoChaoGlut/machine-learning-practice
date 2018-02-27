import arrow
import requests


# 2016-11-14  2017-07-31
def byDay(startTime, endTime):
    url = "http://mrd.meirenji.cn/customer/traffic/querylist/selectData"
    headers = {
        "Host": "mrd.meirenji.cn",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:58.0) Gecko/20100101 Firefox/58.0",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "http://mrd.meirenji.cn/customer/traffic/querylist",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Length": "149",
        "Cookie": "mrj-web-cookike=10213100-f6f7-45ea-8293-f8014660345d",
        "Connection": "keep-alive"
    }
    data = {
        "kpi": "in",
        "dim": "day",
        "startTime": startTime,
        "endTime": endTime,
        "zoneName": "FivePlus-正佳店",
        "instanceIds": "505f904174d64517bbf9c2c115802bd0",
        "chooseType": 0
    }
    response = requests.post(url=url, headers=headers, data=data)
    print(response.json())


def byHour(startTime):
    url = "http://mrd.meirenji.cn/customer/traffic/querylist/selectData"
    headers = {
        "Host": "mrd.meirenji.cn",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:58.0) Gecko/20100101 Firefox/58.0",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "http://mrd.meirenji.cn/customer/traffic/querylist",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Length": "149",
        "Cookie": "mrj-web-cookike=10213100-f6f7-45ea-8293-f8014660345d",
        "Connection": "keep-alive"
    }
    data = {
        "kpi": "in",
        "dim": "hour",
        "startTime": startTime,
        "endTime": "",
        "zoneName": "FivePlus-正佳店",
        "instanceIds": "505f904174d64517bbf9c2c115802bd0",
        "chooseType": 0
    }

    response = requests.post(url=url, headers=headers, data=data)
    print(response.text)


def byHour():
    startTime = arrow.get("2016-11-14", "YYYY-MM-DD")
    endTime = arrow.get("2017-07-31", "YYYY-MM-DD")
    while startTime <= endTime:
        byHour(startTime.format('YYYY-MM-DD'))
        startTime = startTime.shift(days=+1)


if __name__ == '__main__':
    byDay("2016-11-14", "2017-07-31")
    # byHour()
