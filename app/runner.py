from core.trading_utils import StockDataFetcher, Backtest
from app.strategy_loader import load_strategy_class

def run_backtest_on_all(file_path, symbols, intervals, date_range, strategy_args=None,backtest_args=None):

    strategy_args = strategy_args or {}
    start_date, end_date = date_range
    StrategyClass = load_strategy_class(file_path)
    strategy = StrategyClass(**strategy_args)

    results = []

    for symbol in symbols:
        for interval in intervals:
            fetcher = StockDataFetcher(symbol, start_date, end_date, interval)
            df = fetcher.fetch_data()
            df_signals = strategy.apply_strategy(df)

            bt = Backtest(
                trades_df=df_signals,
                fetcher=fetcher,
                initial_balance=backtest_args.get("initial_balance", 10000),
                risk_balance=0.01,
                commission=0.045,
                spread=0.0002
            )

            bt.run_backtest()

            result = {
                "symbol": symbol,
                "interval": interval,
                "balance": bt.balance,
                "win_rate": bt.win_trades / bt.trades_executed if bt.trades_executed else 0,
                "drawdown": bt.max_drawdown,
                "balance_chart": bt.plot_performance(return_fig=True),
                "trade_chart": bt.plot_trades(return_fig=True),
                "trades_table": bt.trade_results.copy()
            }

            results.append(result)

    return results
