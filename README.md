# Robinhood Stock Analyzer

A Python program that analyzes stock data using the Robinhood API.

## Features

- Fetch historical stock data from Robinhood
- Get fundamental data for stocks
- Analyze volatility metrics
- Calculate moving averages and generate trading signals
- Calculate Relative Strength Index (RSI)
- Analyze correlation between different stocks
- Generate comprehensive analysis reports

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/robinhood-stock-analyzer.git
cd robinhood-stock-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main program:
```bash
python stock_analyzer.py
```

2. Enter the stock symbols you want to analyze when prompted (comma-separated, e.g., AAPL,MSFT,GOOGL)

3. The program will:
   - Fetch historical data for the specified stocks
   - Get fundamental data
   - Perform various analyses
   - Generate visualizations and reports
   - Save all results to the `analysis_results` directory

## Authentication

The program supports two authentication methods:

1. **Username/Password Authentication**: The simplest method using robin_stocks library
2. **API Key Authentication**: For advanced usage with the Robinhood API

On first run, you'll be prompted to enter your credentials, which will be saved to a `config.ini` file.

### Generating API Keys

If you want to use API authentication:

1. Run the key generation script:
```bash
python -c "from auth import generate_key_pair; generate_key_pair()"
```

2. Use the public key when creating API credentials in the Robinhood API Credentials Portal
3. Store your API key and private key securely

## Analysis Types

### Volatility Analysis
- Daily volatility
- Annualized volatility
- High-Low range volatility

### Moving Average Analysis
- Short and long-term moving averages
- Buy/sell signals based on crossovers
- Visualization of signals

### RSI Analysis
- Relative Strength Index calculation
- Overbought/oversold identification
- RSI visualization

### Correlation Analysis
- Correlation matrix between different stocks
- Heatmap visualization

## Examples

After running the analysis, you'll find several files in the `analysis_results` directory:

- CSV files with raw analysis data
- PNG images with visualizations
- A summary report in Markdown format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. It is not financial advice, and you should not make investment decisions based solely on the output of this program. Always do your own research and consider consulting a licensed financial advisor before making investment decisions.

The use of this software with the Robinhood API is subject to Robinhood's terms of service and API usage policies.
