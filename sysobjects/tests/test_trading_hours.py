import datetime
from sysobjects.production.trading_hours.trading_hours import tradingHours, listOfTradingHours

def test_next_opening_time():
    now = datetime.datetime.now()
    t1 = tradingHours(now - datetime.timedelta(hours=5), now - datetime.timedelta(hours=2))
    t2 = tradingHours(now + datetime.timedelta(hours=3), now + datetime.timedelta(hours=8))
    t3 = tradingHours(now + datetime.timedelta(hours=10), now + datetime.timedelta(hours=12))
    
    hours_list = listOfTradingHours([t1, t2, t3])
    assert hours_list.next_opening_time() == t2.opening_time

def test_next_opening_time_none_when_past():
    now = datetime.datetime.now()
    t1 = tradingHours(now - datetime.timedelta(hours=5), now - datetime.timedelta(hours=2))
    hours_list = listOfTradingHours([t1])
    assert hours_list.next_opening_time() is None

def test_next_opening_time_empty():
    hours_list = listOfTradingHours([])
    assert hours_list.next_opening_time() is None
