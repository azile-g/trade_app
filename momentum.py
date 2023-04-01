import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math
import base64
from io import BytesIO
from datetime import datetime

def fetch_info():
    try:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        #  Send GET request
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        #  Get the symbols table
        tables = soup.find_all('table')
        #  #  Convert table to dataframe
        df = pd.read_html(str(tables))[1]
        #  Cleanup
        df.drop(columns=['Notes'], inplace=True)
        return df
    except:
        print('Error loading data')
        return None
    
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df, filename = ""):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val) 
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}_{datetime.today().date()}_download.xlsx">Download excel file {filename}</a>'

class IndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.
        Args:
            series(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.
        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with "Nans" values"""
    df = df.copy()
    number_cols = df.select_dtypes("number").columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


def _sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def _ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def _get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)

# https://github.com/bukosabino/ta/blob/master/ta/volatility.py
class BollingerBands(IndicatorMixin):
    """Bollinger Bands
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ):
        self._close = close
        self._window = window
        self._window_dev = window_dev
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        self._mavg = self._close.rolling(self._window, min_periods=min_periods).mean()
        self._mstd = self._close.rolling(self._window, min_periods=min_periods).std(
            ddof=0
        )
        self._hband = self._mavg + self._window_dev * self._mstd
        self._lband = self._mavg - self._window_dev * self._mstd

    def bollinger_mavg(self) -> pd.Series:
        """Bollinger Channel Middle Band
        Returns:
            pandas.Series: New feature generated.
        """
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name="mavg")

    def bollinger_hband(self) -> pd.Series:
        """Bollinger Channel High Band
        Returns:
            pandas.Series: New feature generated.
        """
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name="hband")

    def bollinger_lband(self) -> pd.Series:
        """Bollinger Channel Low Band
        Returns:
            pandas.Series: New feature generated.
        """
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name="lband")

    def bollinger_wband(self) -> pd.Series:
        """Bollinger Channel Band Width
        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width
        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def bollinger_pband(self) -> pd.Series:
        """Bollinger Channel Percentage Band
        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce
        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def bollinger_hband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing High Band (binary).
        It returns 1, if close is higher than bollinger_hband. Else, it returns 0.
        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(
            np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name="bbihband")

    def bollinger_lband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing Low Band (binary).
        It returns 1, if close is lower than bollinger_lband. Else, it returns 0.
        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(
            np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="bbilband")

# https://github.com/bukosabino/ta/blob/master/ta/momentum.py
class RSIIndicator(IndicatorMixin):
    """Relative Strength Index (RSI)
    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.
    https://www.investopedia.com/terms/r/rsi.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        up_direction = diff.where(diff > 0, 0.0)
        down_direction = -diff.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._window
        emaup = up_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        emadn = down_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        relative_strength = emaup / emadn
        self._rsi = pd.Series(
            np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
            index=self._close.index,
        )

    def rsi(self) -> pd.Series:
        """Relative Strength Index (RSI)
        Returns:
            pandas.Series: New feature generated.
        """
        rsi_series = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi_series, name="rsi")
    
    def trading_signals(self, rsi_df, lower, upper) -> pd.DataFrame:
        ## Initialize the columns
        rsi_df['Long Tomorrow'] = np.nan
        rsi_df['Buy Signal'] = np.nan
        rsi_df['Sell Signal'] = np.nan
        rsi_df['Buy RSI'] = np.nan
        rsi_df['Sell RSI'] = np.nan
        rsi_df['Strategy'] = np.nan

        ## Calculate the buy & sell signals
        for x in range(15, len(rsi_df)):
            # Calculate "Long Tomorrow" column
            if ((rsi_df['rsi'][x] <= lower) & (rsi_df['rsi'][x-1]>lower) ):
                rsi_df['Long Tomorrow'][x] = True
            elif ((rsi_df['Long Tomorrow'][x-1] == True) & (rsi_df['rsi'][x] <= upper)):
                rsi_df['Long Tomorrow'][x] = True
            else:
                rsi_df['Long Tomorrow'][x] = False
                
            # Calculate "Buy Signal" column
            if ((rsi_df['Long Tomorrow'][x] == True) & (rsi_df['Long Tomorrow'][x-1] == False)):
                rsi_df['Buy Signal'][x] = rsi_df['Adj Close'][x]
                rsi_df['Buy RSI'][x] = rsi_df['rsi'][x]
                
            # Calculate "Sell Signal" column
            if ((rsi_df['Long Tomorrow'][x] == False) & (rsi_df['Long Tomorrow'][x-1] == True)):
                rsi_df['Sell Signal'][x] = rsi_df['Adj Close'][x]
                rsi_df['Sell RSI'][x] = rsi_df['rsi'][x]
                
        ## Calculate strategy performance?
        rsi_df['Strategy'][15] = rsi_df['Adj Close'][15]
        for x in range(16, len(rsi_df)):
            if rsi_df['Long Tomorrow'][x-1] == True:
                rsi_df['Strategy'][x] = rsi_df['Strategy'][x-1] * (rsi_df['Adj Close'][x] / rsi_df['Adj Close'][x-1])
            else:
                rsi_df['Strategy'][x] = rsi_df['Strategy'][x-1]

        return pd.DataFrame(rsi_df[['Date', 'Long Tomorrow', 'Buy Signal', 'Sell Signal', 'Buy RSI', 'Sell RSI', 'Strategy']])


# https://github.com/bukosabino/ta/blob/master/ta/trend.py    
class MACD(IndicatorMixin):
    """Moving Average Convergence Divergence (MACD)
    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.
    https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = _ema(self._close, self._window_fast, self._fillna)
        self._emaslow = _ema(self._close, self._window_slow, self._fillna)
        self._macd = self._emafast - self._emaslow
        self._macd_signal = _ema(self._macd, self._window_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal

    def macd(self) -> pd.Series:
        """MACD Line
        Returns:
            pandas.Series: New feature generated.
        """
        macd_series = self._check_fillna(self._macd, value=0)
        return pd.Series(
            macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}"
        )

    def macd_signal(self) -> pd.Series:
        """Signal Line
        Returns:
            pandas.Series: New feature generated.
        """

        macd_signal_series = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(
            macd_signal_series,
            name=f"MACD_sign_{self._window_fast}_{self._window_slow}",
        )

    def macd_diff(self) -> pd.Series:
        """MACD Histogram
        Returns:
            pandas.Series: New feature generated.
        """
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(
            macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}"
        )
    
    def trading_signal(self, macd_df, pat):
        macd_col = f"MACD_{self._window_fast}_{self._window_slow}"
        signal_col = f"MACD_sign_{self._window_fast}_{self._window_slow}"
        lst_1 = macd_df[signal_col]
        lst_2 = macd_df[macd_col]
        intersections = []
        signal = []
        day = []
        macd = []
        macd_signal = []
        if len(lst_1) > len(lst_2):
            settle = len(lst_2)
        else:
            settle = len(lst_1)
        for i in range(settle-1):
            if (lst_1[i+1] < lst_2[i+1]) != (lst_1[i] < lst_2[i]):
                if ((lst_1[i+1] < lst_2[i+1]),(lst_1[i] < lst_2[i])) == (True,False):
                    signal.append('buy')
                else:
                    signal.append('sell')
                intersections.append(i)
                day.append(macd_df["Date"].tolist()[i])
                macd.append(macd_df[macd_col].tolist()[i])
                macd_signal.append(macd_df[signal_col].tolist()[i])
        df = {"Date": day, 
              "MACD": macd, 
              "MACD_signal": macd_signal, 
              "Signal": signal}
        ret_macd_df = pd.DataFrame(data = df)
        profit = 0
        profit_lst = []
        for i in range(len(intersections)-pat):
            index = intersections[i]
            true_trade= None
            if macd_df['Close'][index] <= macd_df['Close'][index+pat]:
                true_trade = 'buy'
            elif macd_df['Close'][index] > macd_df['Close'][index+pat]:
                true_trade = 'sell'
            if true_trade != None:
                if signal[i] == true_trade:
                    profit += abs(macd_df['Close'][index]-macd_df['Close'][index+1]) 
                    profit_lst.append(profit)
                if signal[i] != true_trade:
                    profit += -abs(macd_df['Close'][index]-macd_df['Close'][index+1])
                    profit_lst.append(profit)
        return ret_macd_df, profit_lst