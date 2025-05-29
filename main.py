import streamlit as st
from app.strategy_factory import save_strategy_to_file, generate_strategy_code
from app.strategy_loader import load_strategy_class
from app.runner import run_backtest_on_all
from app.gemini_chat import GeminiChat
from langchain_core.messages import HumanMessage

# ========================
# CONFIGURACIÓN INICIAL
# ========================

API_KEY = st.secrets["api"]["key"]

# Configuración de la página con tema personalizado
st.set_page_config(
    page_title="Trading AI Assistant", 
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la estética
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .strategy-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .results-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>💹 Asistente IA de Trading</h1>
    <p>Crea estrategias con inteligencia artificial y analiza su rendimiento</p>
</div>
""", unsafe_allow_html=True)

# ========================
# SIDEBAR - CONFIGURACIÓN
# ========================

with st.sidebar:
    st.markdown("### ⚙️ Configuración de Estrategia")
    
    with st.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Parámetros de riesgo
        st.markdown("#### 🎯 Gestión de Riesgo")
        risk_reward_ratio = st.slider(
            "Risk/Reward Ratio", 
            min_value=0.5, 
            max_value=5.0, 
            step=0.1, 
            value=2.0,
            help="Relación entre ganancia esperada y pérdida máxima"
        )
        
        stop_loss_pct = st.slider(
            "Stop Loss (%)", 
            min_value=0.1, 
            max_value=20.0, 
            step=0.1, 
            value=2.0,
            help="Porcentaje de pérdida máxima antes de cerrar posición"
        ) / 100
        
        st.markdown("#### 💼 Capital Inicial")
        initial_balance = st.number_input(
            "Balance ($)", 
            min_value=1000, 
            max_value=1000000, 
            step=1000, 
            value=10000,
            format="%d"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Selección de Activos")
    
    with st.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Categorías de activos
        asset_categories = {
            "🏢 Acciones Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"],
            "₿ Criptomonedas": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
            "🏦 Sector Financiero": ["JPM", "BAC", "WFC", "GS"],
            "🛍️ Consumo": ["KO", "PG", "WMT", "HD"]
        }
        
        selected_category = st.selectbox("Categoría de Activos", list(asset_categories.keys()))
        symbols = st.multiselect(
            "Activos", 
            options=asset_categories[selected_category], 
            default=asset_categories[selected_category][:2]
        )
        
        intervals = st.multiselect(
            "⏱️ Timeframes", 
            options=["1d", "1h", "15m", "5m"], 
            default=["1d"],
            help="Marcos temporales para el análisis"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# ========================
# MAIN CONTENT
# ========================

# Sección de estrategia
st.markdown('<div class="strategy-section">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Descripción de Estrategia")
    prompt = st.text_area(
        "Describe tu estrategia en lenguaje natural:",
        height=150,
        placeholder="Ejemplo: Comprar cuando el RSI esté por debajo de 30 y la media móvil de 20 períodos esté por encima del precio...",
        help="Sé específico sobre indicadores técnicos, condiciones de entrada y salida"
    )

with col2:
    st.markdown("### 📅 Período de Análisis")
    date_range_inicio = st.date_input(
        "Inicio del backtest",
        value=None,
        help="Selecciona el período para el inicio del backtest"
    )
    date_range_fin = st.date_input(
        "Fin del backtest",
        value=None,
        help="Selecciona el período para el fin backtest"
    )
    
    try:
        days_diff = (date_range_fin - date_range_inicio).days
        st.info(f"📊 Período: {days_diff} días")
    except TypeError:
        st.warning("⚠️ Por favor selecciona un rango de fechas válido.")
    except ValueError:
        st.warning("⚠️ Asegúrate de que la fecha de inicio sea anterior a la fecha de fin.")

st.markdown('</div>', unsafe_allow_html=True)

# Botón de ejecución centrado
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_backtest = st.button("🚀 Ejecutar Backtest", use_container_width=True)

# ========================
# EJECUCIÓN Y RESULTADOS
# ========================

if run_backtest and prompt and symbols and date_range_inicio and date_range_fin:
    
    # Progress bar personalizada
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Paso 1: Generar código
        status_text.text("🧠 Generando estrategia con IA...")
        progress_bar.progress(25)
        
        code = generate_strategy_code(API_KEY, prompt)
        code = code.replace("import pandas as pd", "").replace("import numpy as np", "")
        name = "Estrategia_" + str(abs(hash(prompt)) % 100000)
        file_path = save_strategy_to_file(name, code)
        
        # Paso 2: Ejecutar backtest
        status_text.text("📊 Ejecutando backtests...")
        progress_bar.progress(75)
        
        results = run_backtest_on_all(
            file_path=file_path,
            symbols=symbols,
            intervals=intervals,
            date_range=(str(date_range_inicio), str(date_range_fin)),
            strategy_args={"risk_reward_ratio": risk_reward_ratio, "stop_loss_pct": stop_loss_pct},
            backtest_args={"initial_balance": initial_balance}
        )
        
        progress_bar.progress(100)
        status_text.text("✅ ¡Backtest completado!")
        
        # Limpiar progress bar después de un momento
        import time
        time.sleep(1)
        progress_container.empty()

    # ========================
    # RESULTADOS MEJORADOS
    # ========================
    
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown("## 📈 Resultados del Backtest")
    
    # Resumen general
    if results:
        total_balance = sum([r['balance'] for r in results])
        avg_win_rate = sum([r['win_rate'] for r in results]) / len(results)
        max_drawdown = max([r['drawdown'] for r in results])
        
        # Métricas principales en tarjetas
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        with met_col1:
            st.metric(
                "💰 Balance Total", 
                f"${total_balance:,.2f}",
                delta=f"${total_balance - (initial_balance * len(results)):,.2f}"
            )
        
        with met_col2:
            st.metric(
                "📈 Win Rate Promedio", 
                f"{avg_win_rate:.1%}",
                delta=f"{avg_win_rate - 0.5:.1%}"
            )
        
        with met_col3:
            st.metric(
                "📉 Max Drawdown", 
                f"{max_drawdown:.1%}",
                delta=None
            )
        
        with met_col4:
            roi = ((total_balance / (initial_balance * len(results))) - 1) * 100
            st.metric(
                "📊 ROI Total", 
                f"{roi:.1f}%",
                delta=f"{roi:.1f}%"
            )

    # Resultados detallados por activo
    for i, result in enumerate(results):
        
        with st.expander(f"📊 {result['symbol']} - {result['interval']}", expanded=i==0):
            
            # Métricas específicas del activo
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Balance Final", f"${result['balance']:,.2f}")
                st.metric("📈 Win Rate", f"{result['win_rate']:.1%}")
            
            with col2:
                st.metric("📉 Max Drawdown", f"{result['drawdown']:.1%}")
                roi_individual = ((result['balance'] / initial_balance) - 1) * 100
                st.metric("📊 ROI", f"{roi_individual:.1f}%")
            
            with col3:
                trades_count = len(result['trades_table']) if not result['trades_table'].empty else 0
                st.metric("🔄 Total Trades", trades_count)
                if trades_count > 0:
                    avg_trade = result['trades_table']['PnL'].mean() if 'PnL' in result['trades_table'].columns else 0
                    st.metric("💵 Avg Trade", f"${avg_trade:.2f}")
            
            # Gráficos
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.plotly_chart(
                    result['balance_chart'], 
                    key=f"balance_chart_{result['symbol']}_{result['interval']}", 
                    use_container_width=True
                )
            
            with chart_col2:
                st.plotly_chart(
                    result['trade_chart'], 
                    key=f"trade_chart_{result['symbol']}_{result['interval']}", 
                    use_container_width=True
                )
            
            # Tabla de trades
            trades_df = result['trades_table']
            if not trades_df.empty:
                st.markdown("#### 📋 Historial de Trades")
                
                # Filtros para la tabla
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    trade_type_filter = st.selectbox(
                        "Filtrar por tipo:", 
                        ["Todos", "Buy", "Sell"], 
                        key=f"filter_{result['symbol']}_{result['interval']}"
                    )
                
                # Aplicar filtros
                filtered_df = trades_df.copy()
                if trade_type_filter != "Todos":
                    filtered_df = filtered_df[filtered_df['Type'] == trade_type_filter]
                
                # Mostrar tabla con estilo
                st.dataframe(
                    filtered_df[['Datetime', 'Entry', 'Type', 'PnL'] + 
                               [col for col in ['TakeProfit', 'StopLoss'] if col in filtered_df.columns]], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Estadísticas de trades
                if 'PnL' in filtered_df.columns:
                    profitable_trades = len(filtered_df[filtered_df['PnL'] > 0])
                    total_trades = len(filtered_df)
                    win_rate_trades = profitable_trades / total_trades if total_trades > 0 else 0
                    
                    st.info(f"✅ Trades rentables: {profitable_trades}/{total_trades} ({win_rate_trades:.1%})")
            else:
                st.info('📝 No se registraron trades para este activo.')
            
            st.markdown("#### 🧠 Análisis IA")
            chat = GeminiChat(API_KEY)
            metrics = f"""
            Estrategia: {name}
            Descripción: {prompt}
            Activo: {result['symbol']} ({result['interval']})
            Balance Inicial: ${initial_balance:,}
            Balance Final: ${result['balance']:,.2f}
            ROI: {((result['balance'] / initial_balance) - 1) * 100:.1f}%
            Win Rate: {result['win_rate']:.1%}
            Max Drawdown: {result['drawdown']:.1%}
            Total Trades: {len(result['trades_table']) if not result['trades_table'].empty else 0}
            """

            with st.spinner("🤖 Generando análisis..."):
                response = chat.invoke([HumanMessage(content=f"Analiza esta estrategia de trading y proporciona insights detallados:\n{metrics}.\nIncluye recomendaciones de mejora y posibles riesgos.\n No repitas números, enfócate en explicar qué funciona y qué puede mejorar.")])
                response = response.content.strip()
                # Usar un contenedor con estilo para el análisis
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea; margin: 1rem 0;">
                    {response}
                </div>
                """, unsafe_allow_html=True)

elif run_backtest:
    st.warning("⚠️ Por favor completa todos los campos: estrategia, activos y rango de fechas.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "💡 <strong>Trading AI Assistant</strong> | Desarrollado con Streamlit y Gemini AI"
    "</div>", 
    unsafe_allow_html=True
)