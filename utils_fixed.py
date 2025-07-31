import hashlib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance with fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# File paths for persistent storage
USERS_FILE = "users.json"
PORTFOLIOS_FILE = "portfolios.json"

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify a password against its hash"""
    return hash_password(password) == hashed_password

def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_users(users):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception:
        pass

def load_portfolios():
    """Load portfolios from JSON file"""
    try:
        if os.path.exists(PORTFOLIOS_FILE):
            with open(PORTFOLIOS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_portfolios(portfolios):
    """Save portfolios to JSON file"""
    try:
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump(portfolios, f, indent=2)
    except Exception:
        pass

def get_popular_assets():
    """Return dictionary of popular assets with their symbols"""
    popular_assets = {
        # Stocks
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "JNJ": "Johnson & Johnson",
        
        # ETFs
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        "IWM": "iShares Russell 2000 ETF",
        "EFA": "iShares MSCI EAFE ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "GLD": "SPDR Gold Shares",
        
        # Cryptocurrencies
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "ADA-USD": "Cardano",
        "DOT-USD": "Polkadot",
        "LINK-USD": "Chainlink",
        
        # Commodities
        "GLD": "Gold ETF",
        "SLV": "Silver ETF",
        "USO": "Oil ETF",
        "UNG": "Natural Gas ETF",
        
        # Bonds
        "TLT": "Long-term Treasury Bonds",
        "SHY": "Short-term Treasury Bonds",
        "LQD": "Investment Grade Corporate Bonds",
        "HYG": "High Yield Corporate Bonds",
        "TIP": "Treasury Inflation-Protected Securities"
    }
    
    return popular_assets

def fetch_asset_data(symbol):
    """Fetch current asset data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        # Return mock data if yfinance is not available
        return {
            'name': symbol,
            'current_price': 100.0,
            'symbol': symbol
        }
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current price
        hist = ticker.history(period="1d")
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        
        # Get asset info
        try:
            info = ticker.info
            name = info.get('longName', info.get('shortName', symbol))
        except:
            name = symbol
        
        return {
            'name': name,
            'current_price': float(current_price),
            'symbol': symbol
        }
        
    except Exception:
        return None

def calculate_portfolio_metrics(portfolio):
    """Calculate portfolio performance metrics"""
    if not YFINANCE_AVAILABLE:
        # Return mock metrics if yfinance is not available
        return {
            'beta': 1.0,
            'sharpe_ratio': 0.8,
            'max_drawdown': 5.2
        }
    
    try:
        # Get historical data for all assets
        symbols = list(portfolio.keys())
        
        if not symbols:
            return {
                'beta': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Download data for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Get portfolio weights
        total_value = 0
        weights = {}
        
        for symbol, data in portfolio.items():
            asset_info = fetch_asset_data(symbol)
            if asset_info:
                value = data['shares'] * asset_info['current_price']
                total_value += value
                weights[symbol] = value
        
        # Normalize weights
        if total_value > 0:
            for symbol in weights:
                weights[symbol] = weights[symbol] / total_value
        
        # Download historical data
        portfolio_returns = []
        market_returns = []
        
        try:
            # Get S&P 500 as market benchmark
            market_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Calculate portfolio returns
            portfolio_values = pd.Series(index=market_returns.index, dtype=float)
            portfolio_values[:] = 0
            
            for symbol, weight in weights.items():
                try:
                    asset_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not asset_data.empty:
                        asset_returns = asset_data['Close'].pct_change().dropna()
                        # Align dates and add weighted returns
                        aligned_returns = asset_returns.reindex(portfolio_values.index, fill_value=0)
                        portfolio_values += weight * aligned_returns
                except:
                    continue
            
            portfolio_returns = portfolio_values.dropna()
            
            # Calculate metrics
            if len(portfolio_returns) > 30 and len(market_returns) > 30:
                # Align dates
                common_dates = portfolio_returns.index.intersection(market_returns.index)
                if len(common_dates) > 30:
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    market_aligned = market_returns.loc[common_dates]
                    
                    # Beta calculation
                    covariance = np.cov(portfolio_aligned, market_aligned)[0][1]
                    market_variance = np.var(market_aligned)
                    beta = covariance / market_variance if market_variance != 0 else 1.0
                    
                    # Sharpe ratio (assuming 2% risk-free rate)
                    risk_free_rate = 0.02 / 252  # Daily risk-free rate
                    excess_returns = portfolio_aligned - risk_free_rate
                    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
                    
                    # Max drawdown
                    cumulative_returns = (1 + portfolio_aligned).cumprod()
                    rolling_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - rolling_max) / rolling_max
                    max_drawdown = drawdown.min() * 100
                    
                    return {
                        'beta': beta,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': abs(max_drawdown)
                    }
        
        except Exception:
            pass
        
        return {
            'beta': 1.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
    except Exception:
        return {
            'beta': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

def calculate_technical_indicators(symbol, period="6mo"):
    """Calculate technical indicators for a given symbol"""
    if not YFINANCE_AVAILABLE:
        # Return mock data if yfinance is not available
        dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
        mock_data = pd.DataFrame(index=dates)
        mock_data['Close'] = 100 + np.random.randn(len(dates)).cumsum() * 5
        mock_data['MA_20'] = mock_data['Close'].rolling(window=20).mean()
        mock_data['MA_50'] = mock_data['Close'].rolling(window=50).mean()
        mock_data['RSI'] = 50 + np.random.randn(len(dates)) * 10
        mock_data['MACD'] = np.random.randn(len(dates)) * 2
        mock_data['MACD_Signal'] = mock_data['MACD'].rolling(window=9).mean()
        return mock_data.dropna()
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return None
        
        # Calculate moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        return data.dropna()
        
    except Exception:
        return None

def generate_investment_suggestions(portfolio):
    """Generate investment suggestions based on portfolio analysis"""
    suggestions = []
    
    if not portfolio:
        suggestions.append({
            'type': 'opportunity',
            'message': 'Start building your portfolio by adding diversified assets across different sectors and asset classes.'
        })
        return suggestions
    
    # Analyze asset types
    asset_types = {}
    total_value = 0
    
    for symbol, data in portfolio.items():
        asset_type = data['asset_type']
        asset_info = fetch_asset_data(symbol)
        
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            total_value += value
            
            if asset_type in asset_types:
                asset_types[asset_type] += value
            else:
                asset_types[asset_type] = value
    
    if total_value == 0:
        return suggestions
    
    # Calculate asset type percentages
    asset_percentages = {k: (v/total_value)*100 for k, v in asset_types.items()}
    
    # Diversification suggestions
    if len(asset_types) < 3:
        suggestions.append({
            'type': 'diversification',
            'message': f'Consider diversifying across more asset classes. You currently have {len(asset_types)} asset type(s). Consider adding bonds, REITs, or international exposure.'
        })
    
    # Concentration risk
    max_percentage = max(asset_percentages.values()) if asset_percentages else 0
    if max_percentage > 40:
        max_asset_type = max(asset_percentages, key=asset_percentages.get)
        suggestions.append({
            'type': 'rebalancing',
            'message': f'Your portfolio is heavily concentrated in {max_asset_type} ({max_percentage:.1f}%). Consider reducing this allocation to manage risk.'
        })
    
    # Individual asset concentration
    individual_weights = {}
    for symbol, data in portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            individual_weights[symbol] = (value / total_value) * 100
    
    max_individual = max(individual_weights.values()) if individual_weights else 0
    if max_individual > 25:
        max_symbol = max(individual_weights, key=individual_weights.get)
        suggestions.append({
            'type': 'rebalancing',
            'message': f'{max_symbol} represents {max_individual:.1f}% of your portfolio. Consider reducing single-stock concentration risk.'
        })
    
    # Sector analysis suggestions
    if 'Stock' in asset_percentages and asset_percentages['Stock'] > 60:
        suggestions.append({
            'type': 'diversification',
            'message': 'Your portfolio is stock-heavy. Consider adding bonds or other defensive assets for stability during market downturns.'
        })
    
    # Bond allocation suggestion
    if 'Bond' not in asset_types and total_value > 10000:  # For larger portfolios
        suggestions.append({
            'type': 'opportunity',
            'message': 'Consider adding bond exposure (10-30% allocation) to reduce portfolio volatility and provide steady income.'
        })
    
    # International diversification
    has_international = any('VXUS' in symbol or 'EFA' in symbol or 'VEA' in symbol or 'VWO' in symbol 
                           for symbol in portfolio.keys())
    if not has_international and len(portfolio) > 3:
        suggestions.append({
            'type': 'opportunity',
            'message': 'Add international exposure through ETFs like VEA (developed markets) or VWO (emerging markets) for global diversification.'
        })
    
    # Cryptocurrency allocation
    crypto_allocation = asset_percentages.get('Cryptocurrency', 0)
    if crypto_allocation > 10:
        suggestions.append({
            'type': 'rebalancing',
            'message': f'Cryptocurrency allocation is {crypto_allocation:.1f}%. Consider limiting crypto to 5-10% of portfolio due to high volatility.'
        })
    elif crypto_allocation == 0 and len(portfolio) > 5:
        suggestions.append({
            'type': 'opportunity',
            'message': 'Consider a small allocation (2-5%) to cryptocurrency for potential high returns, but be aware of the risks.'
        })
    
    # Commodity exposure
    has_commodities = 'Commodity' in asset_types
    if not has_commodities and len(portfolio) > 4:
        suggestions.append({
            'type': 'opportunity',
            'message': 'Consider adding commodity exposure (GLD for gold, USO for oil) as an inflation hedge.'
        })
    
    # Regular rebalancing reminder
    suggestions.append({
        'type': 'rebalancing',
        'message': 'Review and rebalance your portfolio quarterly to maintain your target asset allocation and risk profile.'
    })
    
    return suggestions[:6]  # Limit to 6 suggestions

def get_market_data(symbol, period="1y"):
    """Get historical market data for analysis"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception:
        return None

def calculate_correlation_matrix(portfolio):
    """Calculate correlation matrix for portfolio assets"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        symbols = list(portfolio.keys())
        
        if len(symbols) < 2:
            return None
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        returns_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
            except:
                continue
        
        if returns_data.empty:
            return None
        
        correlation_matrix = returns_data.corr()
        return correlation_matrix
        
    except Exception:
        return None

def get_asset_fundamentals(symbol):
    """Get fundamental data for an asset"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
        
        return fundamentals
        
    except Exception:
        return None

def calculate_portfolio_value_history(portfolio, period="6mo"):
    """Calculate historical portfolio value"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180 if period == "6mo" else 365)
        
        portfolio_values = pd.Series(dtype=float)
        
        for symbol, data in portfolio.items():
            try:
                asset_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not asset_data.empty:
                    asset_values = asset_data['Close'] * data['shares']
                    if portfolio_values.empty:
                        portfolio_values = asset_values
                    else:
                        portfolio_values = portfolio_values.add(asset_values, fill_value=0)
            except:
                continue
        
        return portfolio_values.dropna()
        
    except Exception:
        return None

def format_currency(value):
    """Format currency values for display"""
    if value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"

def calculate_risk_metrics(portfolio):
    """Calculate additional risk metrics"""
    try:
        symbols = list(portfolio.keys())
        
        if not symbols:
            return {}
        
        # Get portfolio correlation
        correlation_matrix = calculate_correlation_matrix(portfolio)
        
        if correlation_matrix is not None:
            # Calculate average correlation
            corr_values = correlation_matrix.values
            # Get upper triangle values (excluding diagonal)
            upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
            avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
        else:
            avg_correlation = 0
        
        # Calculate concentration risk
        total_value = 0
        individual_weights = []
        
        for symbol, data in portfolio.items():
            asset_info = fetch_asset_data(symbol)
            if asset_info:
                value = data['shares'] * asset_info['current_price']
                total_value += value
        
        for symbol, data in portfolio.items():
            asset_info = fetch_asset_data(symbol)
            if asset_info:
                value = data['shares'] * asset_info['current_price']
                weight = (value / total_value) if total_value > 0 else 0
                individual_weights.append(weight)
        
        # Herfindahl-Hirschman Index for concentration
        hhi = sum(w**2 for w in individual_weights) if individual_weights else 1
        
        return {
            'average_correlation': avg_correlation,
            'concentration_index': hhi,
            'diversification_ratio': 1 / len(symbols) if symbols else 0
        }
        
    except Exception:
        return {
            'average_correlation': 0,
            'concentration_index': 1,
            'diversification_ratio': 0
        }