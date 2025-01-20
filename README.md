# Thematic Investing on S&P 500 Using AI-Powered Theme Extraction

This repository provides a Python-based pipeline for **thematic investing** on the S&P 500. It leverages **LLM-based techniques (OpenAI ChatGPT or similar)** to **classify companies** into emerging themes and then **optimizes portfolios** within each thematic basket to **maximize risk-adjusted returns**.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Architecture](#architecture)  
3. [Features](#features)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
6. [Analysis & Results](#analysis--results)  
   - [Performance Analysis Across Themes](#performance-analysis-across-themes)  
   - [Key Performance Metrics](#key-performance-metrics)  
7. [Limitations](#limitations)  
8. [Future Improvements](#future-improvements)  
9. [Assumptions](#assumptions)  

---

## Project Overview

**Thematic investing** is a strategy that identifies and invests in key mega-trends shaping the future—like AI & ML, Cloud Computing, Clean Energy, etc. This project:

1. **Fetches** the list of S&P 500 companies from Wikipedia.  
2. **Extracts** relevant themes for each company using an **AI-powered classification** approach with large language models (LLMs).  
3. **Creates** thematic baskets—each basket corresponding to a specific theme.  
4. **Optimizes** each thematic basket’s portfolio using **Modern Portfolio Theory (MPT)** to **maximize risk-adjusted returns**.  
5. **Evaluates** performance via risk-adjusted metrics and visualizes results in interactive charts (efficient frontier, cumulative returns, etc.).

---

## Architecture

Below is a high-level flow of how the system operates:

```
                 +--------------------------+
                 |  Fetch S&P 500 data     |
                 |  (Wikipedia Table)      |
                 +-----------+--------------+
                             |
                             v
                 +--------------------------+
                 |  Parallel AI Analysis    |
                 |  (LLM-based theme        |
                 |   extraction)           |
                 +-----------+--------------+
                             |
                 +-----------v------------+  
                 |  Thematic Baskets     |
                 | (AI/ML, Cloud, etc.)  |
                 +-----------+-----------+
                             |
                             v
                 +------------------------+
                 | Portfolio Optimization |
                 |  (Sharpe Max & Risk    |
                 |   Metrics)            |
                 +-----------+-----------+
                             |
                             v
                 +------------------------+
                 | Performance Metrics &  |
                 | Visualizations         |
                 +------------------------+
```

**Key Steps**:
1. **Data Ingestion & Validation**  
2. **Theme Identification** using LLM-based extraction.  
3. **Basket Creation** for each recognized theme.  
4. **Portfolio Optimization** within each theme.  
5. **Performance Evaluation** (Sharpe, Sortino, VaR, CVaR, etc.) and **visualization**.

---

## Features

- **AI-Based Theme Extraction**: Utilizes OpenAI (or similar LLM) to scrape recent news for each company and assign **relevant investing themes**.  
- **Scalable Pipeline**: Built with multiprocessing and caching (Redis) for **faster classification**.  
- **Portfolio Optimization**: Implements **Modern Portfolio Theory** to find **maximum Sharpe Ratio** allocations.  
- **Risk Management**: Calculates advanced risk metrics: **Value at Risk (VaR)**, **Conditional VaR (CVaR)**, **Max Drawdown**, etc.  
- **Interactive Visualizations**: Generates **interactive HTML** charts for the **Efficient Frontier**, **Cumulative Returns**, and **Theme Comparisons**.  
- **Cloud Deployment Stubs**: Hooks for integration with AWS (S3, Lambda, etc.) for enterprise-scale deployment.

---

## Installation & Setup

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/thematic-investing-llm.git
   cd thematic-investing-llm
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:  
   - Create a `.env` file or export environment variables for your **Redis** and **OpenAI** API credentials.  
     ```bash
     REDIS_HOST=your_redis_host
     REDIS_PORT=your_redis_port
     REDIS_PASSWORD=your_redis_password
     OPENAI_API_KEY=your_openai_api_key
     ```
   - If you **do not** have Redis set up, you can remove or skip its references in code (caching steps).

4. **Data Dumps & Charts Folder**:  
   The script creates a `data_dumps` directory for local caching and a subfolder `charts` for saving HTML plots.

---

## Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```
   This will:
   - Fetch the latest S&P 500 companies.  
   - Check for previous classification results in `analysis_progress.pkl`.  
   - If not found (or if you specify `--reload_classes`), it will re-run the **AI classification**.  
   - Optimize thematic portfolios (or load from `optimized_portfolios.json` if available).  
   - Visualize **Efficient Frontiers**, **Theme Comparisons**, and **Cumulative Returns** in `data_dumps/charts/`.

2. **Command-Line Arguments**:
   - `--reload_classes`: Force re-run LLM classification for each company.  
   - `--reload_optimisation`: Force re-run portfolio optimization (overwrites old results).  
   
   **Example**:
   ```bash
   python main.py --reload_classes --reload_optimisation
   ```

3. **Outputs**:
   - `thematic_baskets.json`: List of themes and their constituent tickers.  
   - `optimized_portfolios.json`: Optimized weights and risk metrics for each theme.  
   - `monitor_performance.json`: Performance metrics for each optimized portfolio.  
   - HTML files in `data_dumps/charts` containing interactive graphs:
     - **Efficient Frontier** per theme.
     - **Theme Comparison** bar charts.
     - **All Themes Cumulative Returns** line plot.

---

## Analysis & Results

### Performance Analysis Across Themes

Below is a **sampled** performance analysis illustrating how thematic portfolios performed from **March 2024 to January 2025**. (Actual results depend on data availability and market conditions.)

1. **Cumulative Returns**  
   - **Cybersecurity** achieved the **highest growth** (*1.80x* of the initial investment), closely followed by **Robotics** and **Cloud Computing** (1.75x).  
   - **AI & ML** performed moderately well with *1.62x*, slightly below **Clean Energy** (1.70x).  
   - **Biotech** underperformed at *1.47x*, suggesting either market headwinds or higher volatility.

2. **Risk-Adjusted Performance** (Sharpe, Sortino, Max Drawdown)  
   - **Sharpe Ratio**:
     - **AI & ML**: *5.60* (highest risk-adjusted returns).  
     - **Biotech**: *3.47* (lowest of the group).  
   - **Sortino Ratio** (focus on downside risk):
     - **AI & ML**: *8.13*, indicating strong upside potential with limited downside.  
     - **Biotech** & **Cybersecurity**: lower ratios, more susceptible to drawdowns.  
   - **Max Drawdown**:
     - **AI & ML**: *-3.5%*, the most resilient.  
     - **Cloud Computing**: *-6.2%*, largest peak-to-trough loss.

3. **Risk Metrics** (VaR and CVaR)  
   - **Value at Risk (VaR)** and **Conditional VaR (CVaR)** are **slightly higher** for high-growth themes like **Cybersecurity** and **Robotics**, indicating **increased risk exposure** to match their returns.

#### Key Takeaways
- **Cybersecurity**: Excellent cumulative return but higher downside risk.  
- **AI & ML**: Top-tier risk-adjusted performance; ideal for investors seeking **balance** between return and risk.  
- **Biotech**: Lagging theme; may need a **re-evaluation** of weighting in the portfolio.  
- **Clean Energy & Robotics**: Stable growth with acceptable risk—good **moderate-risk** picks.

#### Recommendations
1. **Increase Exposure** to **Cybersecurity** (for aggressive) or **AI & ML** (for balanced).  
2. **Reduce Weight** in **Biotech** due to persistent underperformance.  
3. **Watch Volatility** in **Cloud Computing & Metaverse**; consider hedging strategies.

---

### Key Performance Metrics

1. **Sharpe Ratio**  
   - Indicates **risk-adjusted returns** (both upside and downside).  
   - Higher is better.  

2. **Sortino Ratio**  
   - Similar to Sharpe but only penalizes **downside volatility**.  
   - Great for risk-averse perspectives.

3. **Max Drawdown**  
   - Largest observed dip from peak value to trough in a timeframe.  
   - Lower (less negative) is better.

4. **Value at Risk (VaR)**  
   - Maximum expected loss at a certain confidence level (e.g., 95%).  
   - Lower indicates fewer severe losses in normal markets.

5. **Conditional VaR (CVaR)**  
   - Average loss **beyond** VaR (i.e., in the worst cases).  
   - Lower indicates less tail risk.

---

## Limitations

1. **LLM Context Accuracy**: AI theme classification depends on **Google News** scraping; **irrelevant articles** or **inconsistent data** may affect accuracy.  
2. **Limited Historical Data**: The pipeline uses **1 year** of data from **Yahoo Finance**; **longer horizons** or intraday data may be needed for deeper insight.  
3. **Unstructured News Data**: Title-based classification might lack depth; certain themes may not be identified if **headlines** are vague.  
4. **Real-Time Updates**: The approach is **batch-based** and not intended for **high-frequency** or immediate intraday trading signals.

---

## Future Improvements

1. **Advanced NLP & Semantic Filtering**: Integrate more **robust language models** and **embedding-based** semantic filters to reduce irrelevant news.  
2. **Expand Themes Automatically**: Dynamically discover new or trending themes (e.g., **Quantum Computing**, **Space Exploration**) via **clustering**.  
3. **Sentiment Analysis**: Incorporate **positive/negative** news sentiment to further refine the weighting of companies.  
4. **Multi-Year Data**: Expand historical range to detect **longer-term** performance trends.  
5. **Automated Rebalancing**: Implement periodic rebalancing rules, factoring in changes in **volatility** and **theme momentum**.  
6. **Cloud-Native Deployment**: Containerize and deploy on AWS or other cloud platforms for fully **serverless** scaling and scheduled runs.

---

## Assumptions

1. **Reliable API Data**: Assumes **Yahoo Finance** and **Google News** APIs provide accurate, timely data.  
2. **Efficient Market Hypothesis**: The portfolio optimization uses **Modern Portfolio Theory** which assumes markets are relatively efficient.  
3. **No Transaction Costs**: For simplicity, **transaction fees** or **slippage** are not included.  
4. **Stable Risk-Free Rate**: The risk-free rate is **assumed to be constant** (0.02 or 2%) throughout the calculations.

---


**Thank you for exploring this project!**  
