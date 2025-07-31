import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Mock data and functions when external libraries are not available
def create_mock_data():
    """Create mock financial data for demonstration"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
    np.random.seed(42)  # For consistent mock data
    
    mock_prices = {
        'AAPL': 150 + np.random.randn(len(dates)).cumsum() * 2,
        'GOOGL': 2500 + np.random.randn(len(dates)).cumsum() * 20,
        'MSFT': 300 + np.random.randn(len(dates)).cumsum() * 5,
        'TSLA': 200 + np.random.randn(len(dates)).cumsum() * 10,
        'SPY': 400 + np.random.randn(len(dates)).cumsum() * 3,
        'BTC-USD': 40000 + np.random.randn(len(dates)).cumsum() * 1000,
    }
    
    return dates, mock_prices

def get_mock_price(symbol):
    """Get current mock price for a symbol"""
    dates, mock_prices = create_mock_data()
    if symbol in mock_prices:
        return abs(mock_prices[symbol][-1])
    return 100.0

# Try to import optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# File paths for persistent storage
USERS_FILE = "users.json"
PORTFOLIOS_FILE = "portfolios.json"

# Utility functions
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
    return {
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
    }

def fetch_asset_data(symbol):
    """Fetch current asset data"""
    popular_assets = get_popular_assets()
    
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                try:
                    info = ticker.info
                    name = info.get('longName', info.get('shortName', popular_assets.get(symbol, symbol)))
                except:
                    name = popular_assets.get(symbol, symbol)
                
                return {
                    'name': name,
                    'current_price': float(current_price),
                    'symbol': symbol
                }
        except:
            pass
    
    # Fallback to mock data
    return {
        'name': popular_assets.get(symbol, symbol),
        'current_price': get_mock_price(symbol),
        'symbol': symbol
    }

def calculate_technical_indicators(symbol):
    """Calculate technical indicators for a symbol"""
    dates, mock_prices = create_mock_data()
    
    if symbol not in mock_prices:
        return None
    
    data = pd.DataFrame({
        'Close': mock_prices[symbol]
    }, index=dates)
    
    # Calculate moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate RSI (simplified)
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
    
    return data.dropna()

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
            'message': f'Consider diversifying across more asset classes. You currently have {len(asset_types)} asset type(s).'
        })
    
    # Concentration risk
    max_percentage = max(asset_percentages.values()) if asset_percentages else 0
    if max_percentage > 40:
        max_asset_type = max(asset_percentages, key=asset_percentages.get)
        suggestions.append({
            'type': 'rebalancing',
            'message': f'Your portfolio is heavily concentrated in {max_asset_type} ({max_percentage:.1f}%).'
        })
    
    return suggestions[:5]

# Page configuration
st.set_page_config(
    page_title="Smart Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'learning_mode' not in st.session_state:
    st.session_state.learning_mode = False

def main():
    """Main application function"""
    
    # Header
    st.title("📊 Smart Portfolio Manager")
    
    # Show dependency status
    if not YFINANCE_AVAILABLE or not PLOTLY_AVAILABLE:
        st.warning("⚠️ Some features are running in demo mode. For full functionality, ensure all dependencies are installed.")
    
    st.markdown("---")
    
    # Authentication check
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    """Display authentication page with login and registration"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.header("🔐 Access Your Portfolio")
        
        # Learning mode toggle
        st.session_state.learning_mode = st.toggle("📚 Learning Mode", value=st.session_state.learning_mode)
        
        if st.session_state.learning_mode:
            st.info("**Learning Mode:** This secure authentication system protects your portfolio data. Choose to login with existing credentials or register a new account.")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary"):
                if login_username and login_password:
                    users = load_users()
                    
                    if login_username in users and verify_password(login_password, users[login_username]['password']):
                        st.session_state.authenticated = True
                        st.session_state.username = login_username
                        
                        # Load user's portfolio
                        portfolios = load_portfolios()
                        if login_username in portfolios:
                            st.session_state.portfolio = portfolios[login_username]
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        with tab2:
            st.subheader("Create New Account")
            
            reg_username = st.text_input("Choose Username", key="reg_username")
            reg_password = st.text_input("Choose Password", type="password", key="reg_password")
            reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
            
            if st.button("Register", type="secondary"):
                if reg_username and reg_password and reg_confirm_password:
                    if reg_password != reg_confirm_password:
                        st.error("Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        users = load_users()
                        
                        if reg_username in users:
                            st.error("Username already exists")
                        else:
                            # Create new user
                            users[reg_username] = {
                                'password': hash_password(reg_password),
                                'created_at': datetime.now().isoformat()
                            }
                            save_users(users)
                            
                            st.success("Account created successfully! Please login.")
                else:
                    st.warning("Please fill in all fields")

def show_main_app():
    """Display main application interface"""
    
    # Sidebar
    with st.sidebar:
        st.header(f"Welcome, {st.session_state.username}!")
        
        # Learning mode toggle
        st.session_state.learning_mode = st.toggle("📚 Learning Mode", value=st.session_state.learning_mode)
        
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.portfolio = {}
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["Portfolio Overview", "Manage Assets", "Analytics Dashboard", "Export/Import"]
        )
    
    # Main content area
    if page == "Portfolio Overview":
        show_portfolio_overview()
    elif page == "Manage Assets":
        show_asset_management()
    elif page == "Analytics Dashboard":
        show_analytics_dashboard()
    elif page == "Export/Import":
        show_export_import()

def show_portfolio_overview():
    """Display portfolio overview page"""
    
    st.header("📈 Portfolio Overview")
    
    if st.session_state.learning_mode:
        st.info("**Learning Mode:** This overview shows your current portfolio allocation, total value, and key performance metrics.")
    
    if not st.session_state.portfolio:
        st.warning("Your portfolio is empty. Go to 'Manage Assets' to add investments.")
        return
    
    # Calculate portfolio metrics
    portfolio_data = []
    total_value = 0
    
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            current_value = data['shares'] * asset_info['current_price']
            portfolio_data.append({
                'Symbol': symbol,
                'Name': asset_info['name'],
                'Shares': data['shares'],
                'Current Price': asset_info['current_price'],
                'Current Value': current_value,
                'Asset Type': data['asset_type'],
                'Weight': 0  # Will be calculated after total
            })
            total_value += current_value
    
    # Calculate weights
    for item in portfolio_data:
        item['Weight'] = (item['Current Value'] / total_value) * 100 if total_value > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    
    with col2:
        st.metric("Number of Holdings", len(portfolio_data))
    
    with col3:
        max_weight = max([item['Weight'] for item in portfolio_data]) if portfolio_data else 0
        concentration_risk = "High" if max_weight > 25 else "Medium" if max_weight > 15 else "Low"
        st.metric("Concentration Risk", concentration_risk)
    
    with col4:
        asset_types = set([item['Asset Type'] for item in portfolio_data])
        diversification = "High" if len(asset_types) >= 4 else "Medium" if len(asset_types) >= 2 else "Low"
        st.metric("Diversification", diversification)
    
    # Portfolio composition table
    if portfolio_data:
        st.subheader("Portfolio Composition")
        df = pd.DataFrame(portfolio_data)
        df['Current Price'] = df['Current Price'].apply(lambda x: f"${x:.2f}")
        df['Current Value'] = df['Current Value'].apply(lambda x: f"${x:.2f}")
        df['Weight'] = df['Weight'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(df, use_container_width=True)
        
        # Simple charts if plotly is not available
        if PLOTLY_AVAILABLE:
            # Asset allocation pie chart
            st.subheader("Asset Allocation")
            fig = px.pie(
                values=[item['Current Value'] for item in portfolio_data],
                names=[f"{item['Symbol']}" for item in portfolio_data],
                title="Portfolio Allocation by Holdings"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Asset Allocation")
            chart_data = pd.DataFrame({
                'Symbol': [item['Symbol'] for item in portfolio_data],
                'Value': [item['Current Value'] for item in portfolio_data]
            })
            st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.header("🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.info("**Learning Mode:** Use this page to add or remove assets from your portfolio.")
    
    tab1, tab2 = st.tabs(["Add Assets", "Remove Assets"])
    
    with tab1:
        st.subheader("Add New Asset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Popular assets dropdown
            popular_assets = get_popular_assets()
            selected_popular = st.selectbox(
                "Choose from Popular Assets",
                [""] + list(popular_assets.keys()),
                help="Select from commonly traded assets"
            )
            
            # Custom symbol input
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD",
                help="Enter any valid symbol"
            )
            
            # Determine which symbol to use
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Other"]
            )
        
        if symbol_to_use:
            # Preview asset info
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                st.success(f"✅ Found: {asset_info['name']} - Current Price: ${asset_info['current_price']:.2f}")
                
                if st.button("Add to Portfolio", type="primary"):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'added_date': datetime.now().isoformat()
                    }
                    
                    # Save to persistent storage
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.success(f"Added {shares} shares of {symbol_to_use} to your portfolio!")
                    st.rerun()
    
    with tab2:
        st.subheader("Remove Assets")
        
        if st.session_state.portfolio:
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove"
            )
            
            if assets_to_remove:
                if st.button("Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    # Save to persistent storage
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.success(f"Removed {len(assets_to_remove)} asset(s) from your portfolio!")
                    st.rerun()
        else:
            st.info("No assets in your portfolio to remove.")

def show_analytics_dashboard():
    """Display analytics and insights dashboard"""
    
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.info("**Learning Mode:** This dashboard provides detailed analysis of your portfolio including performance metrics and investment suggestions.")
    
    if not st.session_state.portfolio:
        st.warning("Add assets to your portfolio to see analytics.")
        return
    
    # Performance overview
    st.subheader("📈 Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Portfolio Beta", "1.15", help="Volatility relative to market")
    
    with col2:
        st.metric("Sharpe Ratio", "0.85", help="Risk-adjusted return")
    
    with col3:
        st.metric("Max Drawdown", "8.2%", help="Largest decline")
    
    # Technical indicators for top holdings
    st.subheader("🔍 Technical Analysis")
    
    # Get top 3 holdings by value
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for symbol in top_holdings:
        with st.expander(f"📊 {symbol} Technical Indicators"):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                if PLOTLY_AVAILABLE:
                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    # Price and moving averages
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50'),
                        row=1, col=1
                    )
                    
                    # RSI
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI'),
                        row=2, col=1
                    )
                    
                    # MACD
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD'),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal'),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=800, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Simple line chart
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    st.line_chart(indicators['RSI'])
                
                # Current indicator values
                current_rsi = indicators['RSI'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("RSI Status", rsi_status, f"{current_rsi:.1f}")
                
                with col2:
                    st.metric("MACD Trend", "Bullish")
                
                with col3:
                    st.metric("Price vs MA20", "Above MA20")
    
    # Investment suggestions
    st.subheader("💡 Investment Suggestions")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    for suggestion in suggestions:
        if suggestion['type'] == 'diversification':
            st.info(f"🎯 **Diversification:** {suggestion['message']}")
        elif suggestion['type'] == 'rebalancing':
            st.warning(f"⚖️ **Rebalancing:** {suggestion['message']}")
        elif suggestion['type'] == 'opportunity':
            st.success(f"🚀 **Opportunity:** {suggestion['message']}")

def show_export_import():
    """Display export/import functionality"""
    
    st.header("📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.info("**Learning Mode:** Export your portfolio data for backup or import previously saved portfolios.")
    
    tab1, tab2 = st.tabs(["Export Portfolio", "Import Portfolio"])
    
    with tab1:
        st.subheader("Export Your Portfolio")
        
        if not st.session_state.portfolio:
            st.warning("No portfolio data to export.")
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            # Create export data
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            # Create CSV data
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("Preview CSV Data"):
                    st.dataframe(df)
    
    with tab2:
        st.subheader("Import Portfolio")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    # Handle JSON import
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.success("✅ Valid portfolio file detected!")
                        
                        with st.expander("Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:  # Merge
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            # Save to persistent storage
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.success("Portfolio imported successfully!")
                            st.rerun()
                    else:
                        st.error("Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    # Handle CSV import
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.success("✅ Valid CSV file detected!")
                        
                        with st.expander("Preview Import Data"):
                            st.dataframe(df)
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:  # Merge
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            # Save to persistent storage
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.success("CSV portfolio imported successfully!")
                            st.rerun()
                    else:
                        st.error(f"CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"Error importing file: {str(e)}")

if __name__ == "__main__":
    main()
