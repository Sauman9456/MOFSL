import os
import json
import pickle
import logging
import multiprocessing
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import yfinance as yf
from scipy.optimize import minimize
from typing import Dict, List, Any

from scripts import _process_single_company, themes as theme_list


from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThematicBasketCreator:
    """
    A class that orchestrates the thematic investing pipeline:

    1) Data ingestion for S&P 500.
    2) Advanced AI theme extraction (embeddings, clustering, sentiment).
    3) Basket creation.
    4) Portfolio optimization and risk analysis (VaR, CVaR).
    5) Scalability with multiprocessing and caching.
    6) Cloud deployment stubs (AWS).
    7) Data validation / quality checks.
    8) Performance monitoring (metrics).

    Attributes
    ----------
    data_dir : str
        Directory path for data dumps.
    themes : List[str]
        List of available themes.

    Usage
    -----
    from thematic_basket_creator import ThematicBasketCreator

    creator = ThematicBasketCreator()
    sp500_df = creator.fetch_sp500_data()
    baskets = creator.create_thematic_baskets(sp500_df)
    ...
    """

    def __init__(self):
        """
        Initialize the ThematicBasketCreator instance.

        - Sets up available themes from 'theme_list'.
        - Creates or ensures the existence of a 'data_dumps' directory.
        """
        self.themes = theme_list

        # Create data dumps directory
        self.data_dir = "data_dumps"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}{os.sep}charts", exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Data Ingestion
    # ------------------------------------------------------------------------
    def fetch_sp500_data(self) -> pd.DataFrame:
        """
        Fetch S&P 500 companies from Wikipedia.

        This method downloads the current list of S&P 500 constituents
        from Wikipedia, returning them as a pandas DataFrame with 'Symbol'
        and 'Security' columns, among others.

        Returns
        -------
        pd.DataFrame
            DataFrame of S&P 500 companies with multiple columns including
            'Symbol' and 'Security'.

        Raises
        ------
        ValueError
            If the fetched DataFrame is empty or the URL could not be accessed.
        """

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            df = pd.read_html(url)[0]
            # Basic data validation
            if df.empty:
                raise ValueError("Fetched S&P 500 table is empty.")
            logger.info(f"Successfully fetched {len(df)} S&P 500 companies.")
            return df
        except Exception as e:
            logger.error(f"Error fetching S&P 500 data: {e}")
            raise

    # ------------------------------------------------------------------------
    # 2. Data Validation & Quality Checks
    # ------------------------------------------------------------------------
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Performs basic data quality checks on the provided DataFrame.

        Checks if:
          1) The DataFrame is not None or empty.
          2) Logs any columns with missing values.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Raises
        ------
        ValueError
            If the DataFrame is None or empty.
        """

        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty. Validation failed.")

        # Check missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.info(f"Missing values found:\n{missing}")

    # ------------------------------------------------------------------------
    # 3. Create Thematic Baskets
    # ------------------------------------------------------------------------
    def create_thematic_baskets(
        self,
        sp500_df: pd.DataFrame,
        use_advanced: bool = False,
        reload_classes: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Categorize each S&P 500 company into one or more thematic baskets.

        Steps:
          1) Validates the S&P 500 DataFrame.
          2) If 'reload_classes' is False, attempts to load existing classification
             results from a file (analysis_progress.pkl).
          3) If the classification file does not exist or 'reload_classes' is True,
             it runs the AI classification for each company in parallel.
          4) Stores classification progress in 'analysis_progress.pkl' to handle
             crash recovery.
          5) Aggregates the results into a dictionary: {theme: [list_of_symbols]}.

        Parameters
        ----------
        sp500_df : pd.DataFrame
            DataFrame of S&P 500 constituents.
        use_advanced : bool, optional
            Placeholder flag to indicate using advanced AI approaches (default=False).
        reload_classes : bool, optional
            If True, re-run classification from scratch. Otherwise, attempt
            to load from 'analysis_progress.pkl' (default=False).

        Returns
        -------
        Dict[str, List[str]]
            A mapping of theme names to lists of tickers (e.g., {"AI & ML": ["AAPL", ...]}).
        """

        # Example baskets
        thematic_baskets = thematic_baskets = {item: [] for item in self.themes}

        # Validate the data
        self.validate_data(sp500_df)

        # Progress file for crash-recovery
        progress_file = os.path.join(self.data_dir, "analysis_progress.pkl")
        if not reload_classes and os.path.exists(progress_file):
            logger.info(
                "Loading existing classification results from analysis_progress.pkl."
            )
            with open(progress_file, "rb") as f:
                processed_companies = pickle.load(f)
        else:
            logger.info(
                "Re-running classification (AI technique) on all S&P 500 symbols."
            )
            processed_companies = {}

        # Prepare a list of unprocessed companies
        sp500_list = sp500_df.to_dict("records")
        unprocessed = [
            row for row in sp500_list if row["Symbol"] not in processed_companies
        ]

        # Use pool.map for parallel processing
        # This function calls a helper that returns (symbol, context, themes)
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(_process_single_company, unprocessed)

        # Combine results with existing progress
        for symbol, context, themes in results:
            processed_companies[symbol] = {"context": context, "themes": themes}

        # Save updated progress
        with open(progress_file, "wb") as f:
            pickle.dump(processed_companies, f)

        # Now categorize all companies into baskets
        for symbol, data_dict in processed_companies.items():
            self._categorize_company(symbol, data_dict["themes"], thematic_baskets)

        return thematic_baskets

    def _categorize_company(
        self, symbol: str, themes: List[str], baskets: Dict[str, List[str]]
    ):
        """
        Adds a company's ticker symbol to the corresponding thematic baskets.

        If a theme is found in the company's identified themes list,
        it appends the symbol to that theme's list within the baskets dictionary.

        Parameters
        ----------
        symbol : str
            Ticker symbol for the company.
        themes : List[str]
            A list of themes identified for this company by the AI classifier.
        baskets : Dict[str, List[str]]
            A dictionary mapping theme names to a list of ticker symbols.
        """
        if not themes:
            return

        for theme in baskets.keys():
            # Check if the lowercase version of the key is in the list
            if theme.lower() in [p_themes.lower() for p_themes in themes]:
                # Append symbol to the list of that key
                baskets[theme].append(symbol)

    # ------------------------------------------------------------------------
    # 4. Portfolio Optimization & Risk Management
    # ------------------------------------------------------------------------
    def optimize_basket_portfolio(self, basket: List[str]) -> Dict[str, Any]:
        """
        Optimizes the portfolio weights for a thematic basket to maximize the Sharpe Ratio.

        Steps:
          1) Downloads 1 year of historical closing prices for each symbol in the basket.
          2) Cleans the data (removes tickers with excessive missing data).
          3) Calculates daily returns, annualized returns, and the covariance matrix.
          4) Uses a 'minimize' function from scipy to maximize the Sharpe Ratio
             (equivalent to minimizing the negative Sharpe).
          5) Calculates additional risk metrics like VaR and CVaR.

        Parameters
        ----------
        basket : List[str]
            A list of ticker symbols for a particular theme.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the optimized portfolio's:
              - "weights": Dict[str, float]
              - "sharpe_ratio": float
              - "VaR_95": float
              - "CVaR_95": float

            If optimization fails or the basket is empty, returns an empty dict.
        """
        if len(basket) < 1:
            return {}

        # Fetch historical data
        stock_data = self._download_history_for_basket(basket)

        logger.info(f"[DEBUG] Before dropna: shape={stock_data.shape}")
        logger.info(f"[DEBUG] Missing data each ticker:\n{stock_data.isna().sum()}")

        # Remove any stocks with missing data
        non_na_threshold = (
            0.65 * stock_data.shape[0]
        )  # e.g., keep columns with >=80% data
        stock_data = stock_data.dropna(axis=1, thresh=int(non_na_threshold))

        logger.info(f"[DEBUG] After threshold drop: shape={stock_data.shape}")
        logger.info(f"[DEBUG] Valid columns: {stock_data.columns.tolist()}")

        # Update basket to only include stocks with valid data
        basket = list(stock_data.columns)

        if len(basket) < 2:  # Need at least 2 stocks for optimization
            logger.info("Fewer than 2 valid tickers remain => returning {}")
            return {}

        daily_returns = stock_data.pct_change().dropna()
        annual_returns = daily_returns.mean() * 252
        cov_matrix = daily_returns.cov() * 252

        # Basic MPT optimization
        def portfolio_stats(weights):
            p_return = np.sum(annual_returns * weights)
            p_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return p_return, p_risk

        def neg_sharpe_ratio(weights):
            p_ret, p_risk = portfolio_stats(weights)
            rf = 0.02  # risk-free rate
            return -(p_ret - rf) / p_risk if p_risk != 0 else 0

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in basket]
        init_guess = np.array([1 / len(basket)] * len(basket))

        try:
            # Run optimization
            result = minimize(
                neg_sharpe_ratio,
                init_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if not result.success:
                logger.info(f"Optimization failed for basket: {basket}")
                return {}

            weights = dict(zip(basket, result.x))
            sharpe = -result.fun

            # Collect risk metrics
            simulated_returns = daily_returns.dot(result.x)
            var_95, cvar_95 = self.calculate_var_cvar(simulated_returns)

            return {
                "weights": weights,
                "sharpe_ratio": sharpe,
                "VaR_95": var_95,
                "CVaR_95": cvar_95,
            }
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            return {}

    def calculate_var_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> tuple:
        """
        Calculate the Value at Risk (VaR) and Conditional VaR (CVaR) using
        a historical simulation approach.

        VaR at 95% confidence level means the loss is not expected to exceed
        this value more than 5% of the time. CVaR is the average loss given
        that the loss is worse than VaR.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the portfolio.
        confidence_level : float, optional
            The probability level at which VaR is computed (default=0.95).

        Returns
        -------
        tuple
            (VaR, CVaR) in decimal form (e.g., -0.02 for -2%).
        """
        sorted_returns = np.sort(returns.values)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[index]
        cvar = sorted_returns[:index].mean() if index > 0 else var
        return var, cvar

    def _download_history_for_basket(self, basket: List[str]) -> pd.DataFrame:
        """
        Downloads 1 year of daily historical closing prices from Yahoo Finance
        for all symbols in the given basket.

        Parameters
        ----------
        basket : List[str]
            A list of ticker symbols.

        Returns
        -------
        pd.DataFrame
            DataFrame of daily closing prices with each column being a ticker.
        """
        stock_data = pd.DataFrame()
        for symbol in basket:
            try:
                if symbol == "BRK.B":
                    data = yf.Ticker("BRK-B").history(period="1y")["Close"]
                else:
                    data = yf.Ticker(symbol).history(period="1y")["Close"]
                if not data.empty:
                    stock_data[symbol] = data
            except Exception as e:
                logger.info(f"Error fetching data for {symbol}: {e}")
        return stock_data

    # ------------------------------------------------------------------------
    # 5. Scalability: Cloud Deployment Stubs
    # ------------------------------------------------------------------------
    def setup_cloud_infrastructure(self):
        """
        Placeholder method for AWS integration.

        Example usage:
          - Creates AWS clients for S3, SQS, Lambda if 'boto3' is installed.
          - If 'boto3' is missing or there's a setup error, logs a message.

        Raises
        ------
        ImportError
            If 'boto3' is not installed.
        """
        try:
            import boto3

            self.s3 = boto3.client("s3")
            self.sqs = boto3.client("sqs")
            self.lambda_client = boto3.client("lambda")
            logger.info("AWS clients configured successfully.")
        except ImportError:
            logger.info("boto3 not installed, skipping AWS setup.")
        except Exception as e:
            logger.error(f"Failed to set up cloud infrastructure: {e}")

    # ------------------------------------------------------------------------
    # 6. Visualization: (A) Efficient Frontier
    # ------------------------------------------------------------------------

    def visualize_efficient_frontier(
        self, theme: str, basket: List[str], n_portfolios: int = 2000
    ) -> None:
        """
        Generate and plot random portfolios for the given basket to visualize
        the Efficient Frontier using Plotly, highlighting the max Sharpe ratio.

        Steps:
          1) Fetches historical data for the basket.
          2) Computes daily returns and covariance matrix.
          3) Creates random allocations for 'n_portfolios' portfolios.
          4) Calculates Return, Volatility, and Sharpe Ratio for each.
          5) Plots these in a scatter chart with color=Sharpe Ratio.
          6) Marks the best (max Sharpe) portfolio in red.

        Parameters
        ----------
        theme : str
            Name of the theme for labeling charts.
        basket : List[str]
            List of tickers in the theme's basket.
        n_portfolios : int, optional
            Number of random portfolios to generate (default=2000).

        Returns
        -------
        None
            Saves an interactive HTML file in the 'data_dumps/charts' directory.
        """
        stock_data = self._download_history_for_basket(basket)
        if stock_data.shape[1] < 2:
            logger.info(f"Not enough valid stocks in basket '{theme}' for frontier.")
            return

        daily_returns = stock_data.pct_change().dropna()
        annual_returns = daily_returns.mean() * 252
        cov_matrix = daily_returns.cov() * 252

        results = []
        # We'll store a separate list for a text label (tooltips) for each portfolio
        hover_texts = []

        for _ in range(n_portfolios):
            weights = np.random.random(len(basket))
            weights /= np.sum(weights)
            p_ret = np.sum(weights * annual_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            rf = 0.02  # risk-free rate
            sharpe = (p_ret - rf) / p_vol if p_vol != 0 else 0

            # Build a string of top holdings for tooltip
            top_holdings = []
            for ticker, w in sorted(
                zip(basket, weights), key=lambda x: x[1], reverse=True
            ):
                top_holdings.append(f"{ticker}: {w * 100:.2f}%")

            # If you only want top 3 holdings, you could do:
            top_holdings = top_holdings[:5]

            # Create a single multiline string
            holdings_str = "<br>".join(top_holdings)

            # Add Return, Vol, Sharpe to the text if desired
            # We'll store it in a combined text format
            text_for_hover = (
                f"<b>Return:</b> {p_ret:.2%}<br>"
                f"<b>Volatility:</b> {p_vol:.2%}<br>"
                f"<b>Sharpe:</b> {sharpe:.2f}<br>"
                f"<b>Holdings:</b><br>{holdings_str}"
            )

            hover_texts.append(text_for_hover)
            results.append((p_ret, p_vol, sharpe, weights))

        df_portfolios = pd.DataFrame(
            results, columns=["Return", "Volatility", "Sharpe", "Weights"]
        )
        max_sharpe_idx = df_portfolios["Sharpe"].idxmax()
        max_sharpe = df_portfolios.iloc[max_sharpe_idx]

        # Plot with Plotly
        fig = go.Figure()

        # Scatter for all random portfolios
        fig.add_trace(
            go.Scatter(
                x=df_portfolios["Volatility"],
                y=df_portfolios["Return"],
                mode="markers",
                marker=dict(
                    color=df_portfolios["Sharpe"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                ),
                # Pass the entire text array
                text=hover_texts,
                # Construct a hovertemplate to show the text
                hovertemplate="%{text}<extra></extra>",
                name="Random Portfolios",
            )
        )

        # Highlight max Sharpe portfolio
        fig.add_trace(
            go.Scatter(
                x=[max_sharpe["Volatility"]],
                y=[max_sharpe["Return"]],
                mode="markers+text",
                text=[
                    (
                        f"<b>Max Sharpe</b><br>"
                        f"Return: {max_sharpe['Return']:.2%}<br>"
                        f"Vol: {max_sharpe['Volatility']:.2%}<br>"
                        f"Sharpe: {max_sharpe['Sharpe']:.2f}"
                    )
                ],
                textposition="bottom center",
                marker=dict(color="red", size=10, symbol="star"),
                hovertemplate="",
                name="Max Sharpe",
            )
        )

        fig.update_layout(
            title=f"Efficient Frontier - Theme: {theme}",
            xaxis_title="Volatility (Std Dev)",
            yaxis_title="Annual Return",
            legend=dict(x=0.8, y=0.05),
            template="plotly_white",
        )

        # Save the figure as an interactive HTML file
        output_html = os.path.join(
            self.data_dir, f"charts\efficient_frontier_{theme}.html"
        )
        pio.write_html(fig, file=output_html, auto_open=False)
        logger.info(
            f"Saved Efficient Frontier chart for theme '{theme}' to: {output_html}"
        )

    # ------------------------------------------------------------------------
    # 6. Visualization: (B) Compare Themes (Sharpe, VaR, CVaR) in One Figure
    # ------------------------------------------------------------------------
    def visualize_theme_comparison(self, optimized_portfolios: Dict[str, Any]) -> None:
        """
        Creates a bar chart comparing Sharpe, VaR, and CVaR across all themes
        that have valid optimization results.

        Parameters
        ----------
        optimized_portfolios : Dict[str, Any]
            Dictionary containing the optimization results for each theme,
            keyed by theme name, and including "sharpe_ratio", "VaR_95", and "CVaR_95".

        Returns
        -------
        None
            Saves an interactive HTML file in 'data_dumps/charts'.
        """
        themes_list = []
        sharpe_values = []
        var_values = []
        cvar_values = []

        # Gather data for each theme
        for theme, data in optimized_portfolios.items():
            if data and "weights" in data:  # ensure valid optimization result
                themes_list.append(theme)
                sharpe_values.append(data["sharpe_ratio"])
                var_values.append(data["VaR_95"])
                cvar_values.append(data["CVaR_95"])

        if not themes_list:
            logger.info("No valid themes found to compare.")
            return

        # Create subplots for better clarity
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Sharpe Ratios", "VaR (95%)", "CVaR (95%)"],
        )

        # 1) Sharpe bar
        fig.add_trace(
            go.Bar(x=themes_list, y=sharpe_values, name="Sharpe"),
            row=1,
            col=1,
        )
        # 2) VaR bar
        fig.add_trace(
            go.Bar(x=themes_list, y=var_values, name="VaR(95%)"),
            row=1,
            col=2,
        )
        # 3) CVaR bar
        fig.add_trace(
            go.Bar(x=themes_list, y=cvar_values, name="CVaR(95%)"),
            row=1,
            col=3,
        )

        fig.update_layout(
            title="Comparison of Optimized Portfolios Across Themes",
            template="plotly_white",
            showlegend=False,
        )

        output_html = os.path.join(self.data_dir, r"charts\theme_comparison.html")
        pio.write_html(fig, file=output_html, auto_open=False)
        logger.info(f"Saved theme comparison bar charts to: {output_html}")

    # ------------------------------------------------------------------------
    # 6. Visualization: (C) Cumulative Returns of Optimized Portfolios
    # ------------------------------------------------------------------------
    def visualize_all_themes_cumulative_returns(
        self, baskets: Dict[str, List[str]], optimized_portfolios: Dict[str, Any]
    ) -> None:
        """
        Plots a single interactive line chart showing the cumulative returns
        for each *optimized* thematic basket over the last year.

        Steps:
          1) For each theme, retrieves the basket's tickers.
          2) Builds the portfolio daily returns using the optimized weights.
          3) Computes cumulative returns and plots them on a single chart.

        Parameters
        ----------
        baskets : Dict[str, List[str]]
            Dictionary mapping theme names to a list of tickers.
        optimized_portfolios : Dict[str, Any]
            Dictionary containing the optimized weights for each theme,
            keyed by theme name.

        Returns
        -------
        None
            Saves an interactive HTML file in 'data_dumps/charts'.
        """
        fig = go.Figure()

        for theme, data in optimized_portfolios.items():
            if not data or "weights" not in data:
                continue  # skip invalid

            # The corresponding basket's tickers
            tickers = baskets.get(theme, [])
            if not tickers:
                continue

            stock_data = self._download_history_for_basket(tickers)
            if stock_data.empty:
                continue

            # reorder columns to match the ordering in data['weights']
            w_dict = data["weights"]
            valid_cols = [
                ticker for ticker in w_dict.keys() if ticker in stock_data.columns
            ]
            if len(valid_cols) < 1:
                continue

            # subset the DataFrame and reorder
            stock_data = stock_data[valid_cols]
            daily_ret = stock_data.pct_change().dropna()

            # create a weight array with the same ordering
            weight_array = np.array([w_dict[t] for t in valid_cols])
            # portfolio daily returns
            portfolio_daily_ret = daily_ret.dot(weight_array)
            # cumulative returns (start at 1)
            portfolio_cum_returns = (1 + portfolio_daily_ret).cumprod()

            fig.add_trace(
                go.Scatter(
                    x=portfolio_cum_returns.index,
                    y=portfolio_cum_returns.values,
                    mode="lines",
                    name=f"{theme}",
                    hovertemplate=(
                        f"Theme: {theme}<br>"
                        "Date: %{x}<br>"
                        "Cumulative Return: %{y:.2f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Cumulative Returns of Optimized Thematic Portfolios",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template="plotly_white",
            legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)"),
        )

        output_html = os.path.join(self.data_dir, r"charts\all_themes_cum_returns.html")
        pio.write_html(fig, file=output_html, auto_open=False)
        logger.info(f"Saved cumulative returns comparison to: {output_html}")

    # ------------------------------------------------------------------------
    # 7. Performance Monitoring
    # ------------------------------------------------------------------------
    def monitor_performance(self, portfolio_returns: pd.Series):
        """
        Calculate key performance metrics for a given series of daily portfolio returns:
         - Sharpe Ratio
         - Sortino Ratio
         - Maximum Drawdown
         - Tracking Error (placeholder if no benchmark provided)

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily returns of the portfolio.

        Returns
        -------
        Dict[str, float]
            Dictionary containing 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', and 'tracking_error'.
        """

        metrics = {
            "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_returns),
            "sortino_ratio": self._calculate_sortino_ratio(portfolio_returns),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "tracking_error": self._calculate_tracking_error(
                portfolio_returns, benchmark=None
            ),
        }
        logger.info(f"Performance metrics: {metrics}")
        return metrics

    def _calculate_sharpe_ratio(self, returns: pd.Series, rf: float = 0.02) -> float:
        """
        Computes the Sharpe Ratio for daily returns, annualized.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the portfolio or stock.
        rf : float, optional
            The risk-free rate (default=0.02 for 2% annual).

        Returns
        -------
        float
            The annualized Sharpe Ratio. A higher value indicates better
            risk-adjusted performance.
        """
        mean_ret = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)
        if vol == 0:
            return 0
        return (mean_ret - rf) / vol

    def _calculate_sortino_ratio(self, returns: pd.Series, rf: float = 0.02) -> float:
        """
        Computes the Sortino Ratio, focusing only on downside volatility.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the portfolio or stock.
        rf : float, optional
            The risk-free rate (default=0.02).

        Returns
        -------
        float
            The annualized Sortino Ratio, which isolates downside volatility.
        """
        mean_ret = returns.mean() * 252
        negative_vol = returns[returns < 0].std() * np.sqrt(252)
        if negative_vol == 0:
            return 0
        return (mean_ret - rf) / negative_vol

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculates the maximum drawdown for a series of daily returns.

        Steps:
          1) Convert daily returns to cumulative returns.
          2) Track the running maximum of the cumulative returns.
          3) Maximum drawdown is the maximum % difference from the peak.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the portfolio.

        Returns
        -------
        float
            Max drawdown (negative value indicates the percentage drop).
            E.g., -0.10 = -10% from peak.
        """
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()
        return max_dd

    def _calculate_tracking_error(
        self, returns: pd.Series, benchmark: pd.Series = None
    ) -> float:
        """
        Computes the Tracking Error relative to a benchmark.

        If no benchmark is provided or length mismatch,
        returns 0.0 by default.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the portfolio.
        benchmark : pd.Series, optional
            Daily returns of a benchmark.

        Returns
        -------
        float
            The annualized tracking error. A smaller value indicates the
            portfolio more closely follows the benchmark.
        """

        # If no benchmark, just return 0 as placeholder
        if benchmark is None or len(returns) != len(benchmark):
            return 0.0
        diff = returns - benchmark
        return diff.std() * np.sqrt(252)

    # ------------------------------------------------------------------------
    # 8. Documentation & Testing
    # ------------------------------------------------------------------------
    def run_unit_tests(self):
        """
        Placeholder for an internal suite of unit tests that covers:
          - Data ingestion
          - Theme extraction
          - Portfolio optimization
          - Risk metric calculations
        """
        logger.info("Running internal unit tests...")
        # Implement your test cases here
        # e.g., self._test_data_ingestion(), self._test_theme_extraction(), ...
        pass


def parse_arguments():
    """
    Parse command-line arguments for controlling reload behaviors.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed arguments:
         - reload_classes: bool
         - reload_optimisation: bool
    """
    parser = argparse.ArgumentParser(description="Thematic Investing CLI")
    parser.add_argument(
        "--reload_classes",
        default=False,
        action="store_true",
        help="If true, re-run AI classification instead of loading from analysis_progress.pkl.",
    )
    parser.add_argument(
        "--reload_optimisation",
        default=False,
        action="store_true",
        help="If true, re-run portfolio optimization instead of loading from optimized_portfolios.json.",
    )
    return parser.parse_args()


def main():
    """
    Example end-to-end usage of ThematicBasketCreator via terminal:

    1) Parse arguments for reload behavior
    2) Fetch S&P 500
    3) Create thematic baskets
    4) Optimize each basket (or reload from cached results)
    5) Visualize the frontier for all baskets that have at least 2 tickers
    6) Save final results to JSON for usage in a PPT or other business presentations

    Usage:
    ------
    python main.py --reload_classes --reload_optimisation
    """
    args = parse_arguments()

    creator = ThematicBasketCreator()

    # 1) Fetch S&P 500 data
    sp500_df = creator.fetch_sp500_data()

    # 2) Create thematic baskets with potential advanced AI
    baskets = creator.create_thematic_baskets(
        sp500_df, use_advanced=True, reload_classes=args.reload_classes
    )
    logger.info(f"Thematic Baskets discovered:\n{json.dumps(baskets, indent=2)}")

    # 3) Optimize each basket
    #    If --reload_optimisation is False, try to load from JSON if available
    optimized_file = os.path.join(creator.data_dir, "optimized_portfolios.json")
    optimized_portfolios = {}

    if not args.reload_optimisation and os.path.exists(optimized_file):
        logger.info(
            "Loading existing optimization results from optimized_portfolios.json."
        )
        with open(optimized_file, "r") as f:
            optimized_portfolios = json.load(f)
    else:
        logger.info("Re-running portfolio optimization for all themes.")
        for theme, tickers in baskets.items():
            if tickers:
                logger.info(
                    f"Optimizing portfolio for theme: {theme} (Tickers: {len(tickers)})"
                )
                result = creator.optimize_basket_portfolio(tickers)
                optimized_portfolios[theme] = result

        # Save newly optimized results
        with open(optimized_file, "w") as f:
            json.dump(optimized_portfolios, f, indent=2)

    logger.info("Optimization done. Results:")
    logger.info(json.dumps(optimized_portfolios, indent=2))

    # 4) Visualization for all themes
    #    Creates one interactive Efficient Frontier HTML per theme
    for theme, tickers in baskets.items():
        if len(tickers) >= 2:
            creator.visualize_efficient_frontier(theme=theme, basket=tickers)
        else:
            logger.info(
                f"Skipping visualization for theme '{theme}' due to insufficient tickers."
            )

    # 5) Additional visualizations across all themes
    creator.visualize_theme_comparison(optimized_portfolios)
    creator.visualize_all_themes_cumulative_returns(baskets, optimized_portfolios)

    # 6) Performance Monitoring for each theme
    logger.info("Monitoring performance of each theme's optimized portfolio...")

    # Dictionary to hold all performance metrics
    all_performance_metrics = {}

    for theme, result in optimized_portfolios.items():
        if not result or "weights" not in result:
            continue

        tickers = baskets.get(theme, [])
        stock_data = creator._download_history_for_basket(tickers).pct_change().dropna()

        w_dict = result["weights"]
        valid_cols = [t for t in w_dict.keys() if t in stock_data.columns]
        if not valid_cols:
            continue

        # Reorder columns to match the weighting dictionary
        stock_data = stock_data[valid_cols]
        weights = np.array([w_dict[t] for t in valid_cols])
        portfolio_daily_returns = stock_data.dot(weights)

        logger.info(f"Performance metrics for theme: {theme}")
        performance_metrics = creator.monitor_performance(portfolio_daily_returns)

        # Store the metrics for JSON logging
        all_performance_metrics[theme] = performance_metrics

    # Write all performance metrics to JSON
    monitor_file = os.path.join(creator.data_dir, "monitor_performance.json")
    with open(monitor_file, "w") as f:
        json.dump(all_performance_metrics, f, indent=2)
    logger.info(f"All performance metrics saved to {monitor_file}")

    # 5) Save final baskets
    baskets_file = os.path.join(creator.data_dir, "thematic_baskets.json")
    with open(baskets_file, "w") as f:
        json.dump(baskets, f, indent=2)

    logger.info(
        f"End-to-end thematic investing pipeline completed. Results saved under {creator.data_dir}."
    )


if __name__ == "__main__":
    main()
