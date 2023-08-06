# 交易系统，封装回测、优化基本过程


import backtrader as bt
import quantstats
#import akshare as ak
#import efinance as ef
import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import run
import sys
#import math
import imgkit
#from PIL import Image
from scipy import stats
import empyrical as ey
import itertools 
import collections
import datetime




# 设置显示环境
def init_display():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    np.set_printoptions(threshold = sys.maxsize)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    
# 获取数据
# @run.change_dir
def get_data(code, start_date = "2020-01-01", end_date = "2020-12-31", refresh = False):
    def download_data(code):
        try:
            data = yf.download(code, start = start_date, end=end_date)
        except KeyError:
            print(" ".join(["Error in",__file__]))
            # if adjust == "qfq":
            #     fqt = 1
            # elif adjust == "hfq":
            #     fqt = 2
            
            # if period == "daily":
            #     klt = 101
            # elif period == "weekly":
            #     klt = 102
            # elif period == "monthly":
            #     klt = 103
            # data = ef.stock.get_quote_history(code, beg = start_date, end = end_date, fqt = fqt, klt = klt)
        # print(data.keys())
        # data.date = pd.to_datetime(data['Datetime'])
        # data.set_index("date", drop = False, inplace = True)
        return data
            
    stockfile = os.environ['TICKER_DATA']+"/"+code+".csv"
    if os.path.exists(stockfile) and refresh == False:
        # print('Get from local\n')
        stock_data = pd.read_csv(stockfile,
                                 parse_dates=[0])
        
        # stock_data.date = pd.to_datetime(stock_data['Date']) # illegal
        stock_data.set_index("Date", drop = True, inplace = True)
        # stock_data = stock_data.loc[start_date:datetime.datetime(end_date) + datetime.timedelta(days = 1), :]
        # end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        #end_date = end_date.strftime('%Y-%m-%d')

        stock_data = stock_data.loc[start_date:, :]
    else:
        # print('Get from yfinance\n')
        stock_data = download_data(code)
        if os.path.exists(stockfile):
            os.system("rm " + stockfile)
        stock_data.to_csv(stockfile)
        stock_data['Date'] = stock_data.index
        stock_data.set_index("Date", drop = False, inplace = True)

    return stock_data

def PrintResults(results):
    print("Result:")
    for i, v in results.items():
        if(isinstance(v, str)):
            print("%-20s %20s" % (i, v))
        else:
            print("%-20s %20.4f" % (i, v))
    
    
# A股的交易成本:买入交佣金，卖出交佣金和印花税
class CNA_Commission(bt.CommInfoBase):
    params = (('stamp_duty', 0.005), # 印花税率 
              ('commission', 0.0001), # 佣金率 
              ('stocklike', True),   ('commtype', bt.CommInfoBase.COMM_PERC),)
    
    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return size * price * self.p.commission
        elif size < 0:
            return - size * price * (self.p.stamp_duty + self.p.commission)
        else:
            return 0
            
            
# 自定义分析器，记录交易成本数据
class CostAnalyzer(bt.Analyzer):
    def __init__(self):
        self._cost = []
        self.ret = 0.0
        
    def notify_trade(self, trade):
        if trade.justopened or trade.status == trade.Closed:
            self._cost.append(trade.commission)
            
    def stop(self):
        super(CostAnalyzer, self).stop()
        self.ret = np.sum(self._cost)
        
    def get_analysis(self):
        return self.ret

    
# 策略类基类
class Strategy(bt.Strategy):
    def __init__(self):
        pass
        
    def log(self, txt, dt = None):
        if self.params.bprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))
                
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("交易被拒绝/现金不足/取消")
        elif order.status in [order.Completed]: 
            if order.isbuy(): 
                self.log('买单执行,%s, %.2f, %i' % (order.data._name, order.executed.price, order.executed.size))
            elif order.issell(): 
                self.log('卖单执行, %s, %.2f, %i' % (order.data._name, order.executed.price, order.executed.size))
        self.order = None
        
    def notify_trade(self, trade): 
        if trade.isclosed: 
            self.log('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f, 市值 %.2f, 现金 %.2f'%(trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()))
                
    def stop(self):
        for i, d in enumerate(self.datas):
            pos = self.getposition(d).size
            if pos != 0:
                # print("关闭", d._name)
                self.close()
                
    # 交易数量取整
    def downcast(self, amount, lot): 
        return abs(amount//lot*lot)
        
    # 判断是否是最后的交易日
    def is_lastday(self,data): 
        try: 
            next_next_close = data.close[2]
        except IndexError: 
            return True 
        except: 
            print("发生其它错误")
            return False
            
            
# 回测类
class BackTest():
    """
        A股股票策略回测类
        strategy   回测策略
        codes      回测股票代码列表
        start_date 回测开始日期
        end_date   回测结束日期
        bk_code    基准股票代码
        rf         无风险收益率
        start_cash 初始资金
        stamp_duty 印花税率，单向征收
        commission 佣金费率，双向征收
        adjust     股票数据复权方式，qfq或hfq
        period     股票数据周期(日周月)
        refresh    是否更新数据
        bprint     是否输出中间结果
        bdraw      是否作图
        **param   策略参数，用于调参
    """
    def __init__(self, strategy, codes, start_date, end_date, bk_code = "SPY", rf = 0.03, start_cash = 10000000, stamp_duty=0.005, commission=0.0001, adjust = "hfq", period = "daily", refresh = False, bprint = False, bdraw = False, **param):
        self._cerebro = bt.Cerebro()
        self._strategy = strategy
        self._codes = codes
        self._bk_code = bk_code
        self._start_date = start_date
        self._end_date = end_date
        # self._stock_data = stock_data
        # self._bk_data = bk_data
        self._rf = rf
        self._start_cash = start_cash
        self._comminfo = CNA_Commission(stamp_duty=0.005, commission=0.0001)
        self._adjust = adjust
        self._period = period
        self._refresh = refresh
        self._bprint = bprint
        self._bdraw = bdraw
        self._param = param
        self._output = os.environ['ALGO_HOME']+"output/"
        
    # 回测前准备
    def _before_test(self):
        for code in self._codes:
            data = get_data(code = code, 
                            start_date = self._start_date, 
                            end_date = self._end_date,
                            # adjust = self._adjust,
                            # period = self._period, 
                            refresh = self._refresh)
            data = self._datatransform(data, code)
            self._cerebro.adddata(data, name = code)
        self._cerebro.addstrategy(self._strategy, bprint = self._bprint, **self._param)
        self._cerebro.broker.setcash(self._start_cash)
        self._cerebro.broker.addcommissioninfo(self._comminfo)
        
    # 数据转换
    def _datatransform(self, stock_data, code):
        # 生成datafeed
        data = bt.feeds.PandasData(
            dataname=stock_data,
            name=code,
            fromdate=stock_data.index.min(),
            todate=stock_data.index.max(),
            datetime=None, # None: use index
            open='Open',
            high='High',
            low='Low',    
            close='Close',
            volume='Volume',
            openinterest=-1
            )
        return data
    
    # 增加分析器
    def _add_analyzer(self):
        self._cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
        self._cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = "TA")
        self._cerebro.addanalyzer(bt.analyzers.TimeReturn, _name = "TR")
        self._cerebro.addanalyzer(bt.analyzers.SQN, _name = "SQN")
        self._cerebro.addanalyzer(bt.analyzers.Returns, _name = "Returns")
        self._cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name = "TimeDrawDown")
        self._cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=self._rf)
        self._cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='SharpeRatio_A')
        self._cerebro.addanalyzer(CostAnalyzer, _name="Cost")
        
    # 运行回测
    def run(self):
        self._before_test()
        self._add_analyzer()
        self._results = self._cerebro.run()
        results = self._get_results()
        results = results[results.index != "Ticker"]
        return results
        
    # 获取回测结果
    def _get_results(self):
        # 计算基准策略收益率
        self._bk_data = get_data(code = self._bk_code, start_date = self._start_date, end_date = self._end_date, refresh = self._refresh)
        bk_ret = self._bk_data.Close.pct_change()
        bk_ret.fillna(0.0, inplace = True)
    
        if self._bdraw:
            self._cerebro.plot(style = "candlestick")
            plt.savefig(f'{self._output}/backtest_result.jpg')
    
        testresults = self._backtest_result(self._results, bk_ret, rf = self._rf)
        end_value = self._cerebro.broker.getvalue()
        pnl = end_value - self._start_cash

        testresults["Start Cash"] = self._start_cash
        testresults["Start Date"] = self._start_date
        testresults["End Date"] = self._end_date
        testresults["End Value"] = end_value
        testresults["Profit and Loss"] = pnl
        try:
            testresults["benefit-cost ratio"] = pnl/testresults["Transaction Cost"]
        except ZeroDivisionError:
            pass
        testresults["Ticker"] = self._codes
        return testresults
        
    # 计算回测指标
    def _backtest_result(self, results, bk_ret, rf = 0.01):
        # 计算回测指标
        portfolio_stats = results[0].analyzers.getbyname('PyFolio')
        returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_convert(None)
        totalTrade = results[0].analyzers.getbyname("TA").get_analysis()
        sqn = results[0].analyzers.SQN.get_analysis()["sqn"]
        Returns = results[0].analyzers.Returns.get_analysis()
        timedrawdown = results[0].analyzers.TimeDrawDown.get_analysis()
        sharpe = results[0].analyzers.SharpeRatio.get_analysis()
        sharpeA = results[0].analyzers.SharpeRatio_A.get_analysis()
        cost = results[0].analyzers.Cost.get_analysis()
        
        backtest_results = pd.Series()

        backtest_results["Total Yeild"] = Returns["rtot"]
        backtest_results["Average Yeild"] = Returns["ravg"]
        backtest_results["IRR"] = Returns["rnorm"]
        backtest_results["Transaction Cost"] = cost
        backtest_results["SQN"] = sqn
        try:
            backtest_results["Trade Times"] = totalTrade["total"]["total"]
            backtest_results["Profit Times"] = totalTrade["won"]["total"]
            backtest_results["Total Profit"] = totalTrade["won"]["pnl"]["total"]
            backtest_results["Avg Profit"] = totalTrade["won"]["pnl"]["average"]
            backtest_results["Max Profit"] = totalTrade["won"]["pnl"]["max"]
            backtest_results["Loss Times"] = totalTrade["lost"]["total"]
            backtest_results["Total Loss"] = totalTrade["lost"]["pnl"]["total"]
            backtest_results["Avg Loss"] = totalTrade["lost"]["pnl"]["average"]
            backtest_results["Max Loss"] = totalTrade["lost"]["pnl"]["max"]
            
            # 胜率就是成功率，例如投入十次，七次盈利，三次亏损，胜率就是70%。
            # 防止被零除 
            if totalTrade["total"]["total"] == 0: 
                backtest_results["WinRate"] = np.NaN 
            else:
                backtest_results["WinRate"] = totalTrade["won"]["total"]/totalTrade["total"]["total"]
            # 赔率是指盈亏比，例如平均每次盈利30%，平均每次亏损10%，盈亏比就是3倍。
            # 防止被零除
            if totalTrade["lost"]["pnl"]["average"] == 0:
                backtest_results["LostRate"] = np.NaN
            else:
                backtest_results["LostRate"] = totalTrade["won"]["pnl"]["average"]/abs(totalTrade["lost"]["pnl"]["average"])
            # 计算风险指标
            self._risk_analyze(backtest_results, returns, bk_ret, rf = rf)
        except KeyError:
            pass
            
        return backtest_results
        
    # 将风险分析和绘图部分提出来
    def _risk_analyze(self, backtest_results, returns, bk_ret, rf = 0.01):
        prepare_returns = False # 已经是收益率序列数据了，不用再转换了
        # 计算夏普比率
        if returns.std() == 0.0:
            sharpe = 0.0
        else:
            sharpe = quantstats.stats.sharpe(returns = returns, rf = rf)
        # 计算αβ值
        alphabeta = quantstats.stats.greeks(returns, bk_ret, prepare_returns = prepare_returns)
        # 计算信息比率
        info = quantstats.stats.information_ratio(returns, bk_ret, prepare_returns = prepare_returns)
        # 索提比率
        sortino = quantstats.stats.sortino(returns = returns, rf = rf)
        # 调整索提比率
        adjust_st = quantstats.stats.adjusted_sortino(returns = returns, rf = rf)
        # skew值
        skew = quantstats.stats.skew(returns = returns, prepare_returns = prepare_returns)
        # calmar值
        calmar = quantstats.stats.calmar(returns = returns, prepare_returns = prepare_returns)
        # r2值
        r2 = quantstats.stats.r_squared(returns, bk_ret, prepare_returns = prepare_returns)
        backtest_results["Volatility"] = quantstats.stats.volatility(returns = returns, prepare_returns = prepare_returns)
        backtest_results["WinRate"] = quantstats.stats.win_rate(returns = returns, prepare_returns = prepare_returns)
        backtest_results["RiskReturnRatio"] = quantstats.stats.risk_return_ratio(returns = returns, prepare_returns = prepare_returns)
        backtest_results["Sharpe"] = sharpe
        backtest_results["α"] = alphabeta.alpha
        backtest_results["β"] = alphabeta.beta
        backtest_results["IR"] = info
        backtest_results["Sortino"] = sortino
        backtest_results["Adj_Sortino"] = adjust_st
        backtest_results["Skew"] = skew
        backtest_results["calmar"] = calmar
        backtest_results["r2"] = r2
    
        # 最大回撤
        mdd = quantstats.stats.max_drawdown(prices = returns)
        backtest_results["MDD"] = mdd
    
        # 生成回测报告
        if self._bdraw:
        # if True:
            self._make_report(returns = returns, bk_ret = bk_ret, rf = rf)
        
    # 回测报告
    def _make_report(self, returns, bk_ret, rf, filename = "report.jpg", title = "Backtest Result", prepare_returns = False):
        # filename = self._code + filename 
        quantstats.reports.html(returns = returns, benchmark = bk_ret, rf = rf, output=f'{self._output}/stats.html', title=title, prepare_returns = prepare_returns)
        # wkhtmltopdf doesn't suppot in Debian 12
        #imgkit.from_file(f'{self._output}/stats.html', f'{self._output}/{filename}', options = {"xvfb": ""})


                
# 对整个市场的股票进行回测
class Research():
    """
        A股市场回测类
        strategy   回测策略
        bk_code    基准股票
        start_date 回测开始日期
        end_date   回测结束日期
        highprice  筛选股票池的最高股价
        lowprice   筛选股票池的最低股价
        min_len    股票数据最小大小(避免新股等)
        start_cash 初始资金大小
        adjust     数据复权方式
        period     数据周期
        retest     是否重新回测
        refresh    是否更新数据
        bprint     是否输出交易过程
        bdraw      是否作图
        **params   策略参数
    """
    def __init__(self, strategy, bk_code, start_date, end_date, highprice = sys.float_info.max, lowprice = 0.0, min_len = 1, start_cash = 10000000, adjust = "hfq", 
period = "daily", retest = False, refresh = False, bprint = False, bdraw = True, tickerlists = 'test1', **params):
        self._strategy = strategy
        self._start_date = start_date
        self._end_date = end_date
        self._bk_code = bk_code
        self._highprice = highprice
        self._lowprice = lowprice
        self._min_len = min_len
        self._start_cash = start_cash
        self._adjust = adjust
        self._period = period
        self._retest = retest
        self._refresh = refresh
        self._bprint = bprint
        self._bdraw = bdraw
        self._params = params
        self._output = os.environ['ALGO_HOME']+"output/"
        self._tickerlist = tickerlists
        
    # 调用接口
    def run(self):
        self._test()
        if self._bdraw:
            # print("测试", self._results.info())
            self._save(self._results)
        # print("测试2")
        # print(self._results.info())
        return self._results
        
    # 对回测结果画图
    def _draw(self, results):
        results.set_index("Ticker", inplace = True)
        # 绘图
        plt.figure()
        results.loc[:, ["SQN", "α", "β", "Trade Times", "IR", "Sharpe", "IRR", "benefit-cost ratio", "MDD", "Sortino", "WinRate", "LostRate"]].hist(bins = 100, figsize = (40, 20))
        plt.suptitle("Backtest Result")
        plt.savefig(f'{self._output}/market_test.jpg')
    
    def _save(self, results):
        results.to_csv(f'{self._output}/research.csv')
        
    # 执行回测    
    def _test(self):
            
        self._codes = self._make_pool(refresh = self._refresh)
        self._results = pd.DataFrame()
        n = len(self._codes)
        i = 0
        print(f'Backtest {n} tickers')
        print(self._codes)
        for code in self._codes:
            i += 1
            print("Progress:", i/n)
            data = get_data(code = code, 
                start_date = self._start_date, 
                end_date = self._end_date,
                refresh = self._refresh)
            if len(data) <= self._min_len or (data.Close < 0.0).sum() > 0:
                continue
            backtest = BackTest(strategy = self._strategy, codes = [code], bk_code = self._bk_code, start_date = self._start_date, end_date = self._end_date, start_cash = self._start_cash, adjust = self._adjust, period = self._period, refresh = self._refresh, bprint = self._bprint, **self._params)
            res = backtest.run()
            res["Ticker"] = code
            # self._results = self._results.append(res, ignore_index = True)
            self._results = pd.concat([self._results, res], ignore_index = False, axis=1)
            # print(self._results)
            # self._results.append(res, ignore_index = True)
            # print("测试2", res)
        self._results = self._results.T
        self._results.sort_values(by = "IRR", inplace = True, ascending = False)
        self._results = self._results.reset_index()
        return
        
    # 形成股票池
    def _make_pool(self, refresh = True):
        data = pd.DataFrame()
        stockfile = f'{os.environ["ALGO_HOME"]}/tickerlists/{self._tickerlist}.csv'
        if os.path.exists(stockfile) and refresh == False:
            data = pd.read_csv(stockfile, dtype = {"a":str}, names=['Ticker'])
        else:
            print(f'Error: {__file__}')
            # stock_zh_a_spot_df = ak.stock_zh_a_spot()
            # stock_zh_a_spot_df.to_csv(stockfile)
            # data = stock_zh_a_spot_df
        # codes = self._select(data)
        return list(data['Ticker'])

    
    
# 对策略进行参数优化
class OptStrategy:
    """
        策略优化类
        codes      股票代码列表
        bk_code    基准股票代码
        strategy   回测策略
        start_date 回测开始日期
        end_date   回测结束日期
        highprice  筛选股票池的最高股价
        lowprice   筛选股票池的最低股价
        min_len    股票数据最小大小(避免新股等)
        start_cash 初始资金大小
        adjust     数据复权方式
        period     数据周期
        retest     是否重新回测
        refresh    是否更新数据
        bprint     是否输出交易过程
        bdraw      是否作图
        num_params 调参的参数个数
        **params   要调优的参数范围
    """
    def __init__(self, codes, strategy, start_date, end_date, bk_code = "000300", min_len = 1, start_cash = 10000000, adjust = "hfq", period = "daily", retest = False, refresh = False, bprint = False, bdraw = True, num_params = 0, **params):
        self._codes = codes
        self._bk_code = bk_code
        self._strategy = strategy
        self._start_date = start_date
        self._end_date = end_date
        self._min_len = min_len
        self._start_cash = start_cash
        self._adjust = adjust
        self._period = period
        self._retest = retest
        self._refresh = refresh
        self._bprint = bprint
        self._bdraw = bdraw
        self._params = params
        self._num_params = num_params
        self._output = os.environ['ALGO_HOME']+"output/"

                
    # 运行回测
    def run(self):
        self._results = pd.DataFrame()
        optparams = []
        optkeys = list(self._params)[-1]
        # 遍历所有参数，初始化回测类，执行回测
        params = self._get_params()

        for param in params:
            backtest = BackTest(
                strategy = self._strategy, 
                codes = self._codes, 
                start_date = self._start_date, 
                end_date = self._end_date, 
                bk_code = self._bk_code,
                start_cash = self._start_cash,
                adjust = self._adjust, 
                period = self._period,
                refresh = self._refresh, 
                bprint = self._bprint, 
                bdraw = self._bdraw,
                **param[0])
            res = backtest.run()
            param_keys = list(param[0])[-self._num_params:]
            for key in param_keys:
                res[f'Param_{key}'] = param[0][key]

            self._results = pd.concat([self._results, res], ignore_index = False, axis=1)
            # input("按任意键继续")

        self._results.sort_values(by = "IRR", inplace = True, ascending = False, axis=1)
        self._results = self._results.T
        self._results = self._results.reset_index(drop=True)
        self._save(self._results)
        return self._results
                    
    # 工具函数，提取参数要用，照Backtrader的optstrategy写的。
    @staticmethod
    def _iterize(iterable): 
        niterable = list() 
        for elem in iterable: 
            if isinstance(elem, str): 
                elem = (elem,) 
            elif not isinstance(elem, collections.Iterable): 
                elem = (elem,)
            niterable.append(elem) 
        return niterable
                    
    # 分析参数列表，提取参数
    def _get_params(self):
        params = self._params
        optkeys = list(params)
        vals = self._iterize(params.values())
        optvals = itertools.product(*vals)
        okwargs1 = map(zip, itertools.repeat(optkeys), optvals)
        optkwargs = map(dict, okwargs1) 
        it = itertools.product(optkwargs)
        return it
        
    # 对回测结果进行排序
    def sort_results(self, results, key, inplace = True, ascending = False):
        # print(results)
        results.sort_values(by = key, inplace = inplace, ascending = ascending)
        # print("测试", results)
        return results
        
    # 对回测结果画图
    # def _draw(self, results):
    #     # print("测试", results.info())
    #     # results.set_index("股票代码", inplace = True)
    #     # 绘图
    #     plt.figure()
    #     results.loc[:, ["SQN", "α", "β", "Trade Times", "IR", "Sharpe", "IRR", "benefit-cost ratio", "MDD", "Sortino", "WinRate", "LostRate"]].hist(bins = 100, figsize = (40, 20))
    #     plt.suptitle("策略参数优化结果")
    #     plt.savefig(f'{self._output}/params_optimize.jpg')
            
    def _save(self, results):
        results.to_csv(f'{self._output}/params_optimize.csv')

if __name__ == "__main__":
    pass