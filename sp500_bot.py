import yfinance as yf
import pandas as pd
import numpy as np
import ta
import concurrent.futures
import datetime
import os

# Redistributed weights after omitting Macro/Sector and Sentiment
WEIGHT_VALUATION = 0.2857
WEIGHT_GROWTH = 0.2857
WEIGHT_HEALTH = 0.2143
WEIGHT_TECH = 0.2143

def get_sp500_tickers():
    print("Obteniendo la lista de tickers del S&P 500...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    # Limpiar tickers para yfinance (ej. BRK.B -> BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def fetch_data_for_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Fundamental Data
        pe = info.get('trailingPE', np.nan)
        pb = info.get('priceToBook', np.nan)
        eps_growth = info.get('earningsGrowth', np.nan)
        rev_growth = info.get('revenueGrowth', np.nan)
        debt_equity = info.get('debtToEquity', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        
        # Technical Data
        hist = stock.history(period="3mo")
        if len(hist) < 30:
            momentum = np.nan
            rsi = np.nan
        else:
            # Momentum: Retorno de 1 mes (aprox 21 dias de trading)
            current_price = hist['Close'].iloc[-1]
            price_1m_ago = hist['Close'].iloc[-21]
            momentum = (current_price - price_1m_ago) / price_1m_ago
            
            # RSI 14 dias
            rsi_indicator = ta.momentum.RSIIndicator(close=hist['Close'], window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            
        return {
            'Ticker': ticker,
            'PE': pe,
            'PB': pb,
            'EPS_Growth': eps_growth,
            'Rev_Growth': rev_growth,
            'Debt_Equity': debt_equity,
            'ROE': roe,
            'Momentum': momentum,
            'RSI': rsi
        }
    except Exception as e:
        # Silencioso en caso de error para no saturar los logs
        return {'Ticker': ticker}

def normalize_metrics(df):
    """
    Normaliza las métricas al rango [0, 1] usando rangos percentiles empíricos
    tal como se indica en el framework para poder compararlas de forma robusta.
    """
    # Para métricas donde MENOR es mejor, invertimos el rango: 1 - rank
    df['Score_PE'] = 1 - df['PE'].rank(pct=True, na_option='keep')
    df['Score_PB'] = 1 - df['PB'].rank(pct=True, na_option='keep')
    df['Score_Debt_Equity'] = 1 - df['Debt_Equity'].rank(pct=True, na_option='keep')
    
    # Para métricas donde MAYOR es mejor
    df['Score_EPS_Growth'] = df['EPS_Growth'].rank(pct=True, na_option='keep')
    df['Score_Rev_Growth'] = df['Rev_Growth'].rank(pct=True, na_option='keep')
    df['Score_ROE'] = df['ROE'].rank(pct=True, na_option='keep')
    df['Score_Momentum'] = df['Momentum'].rank(pct=True, na_option='keep')
    df['Score_RSI'] = df['RSI'].rank(pct=True, na_option='keep')
    
    # Llenar NaN con 0.5 (neutral) o 0 (penalización). 
    # Para un análisis conservador, usamos 0.5 (neutro) para que falte un dato no hunda totalmente a la empresa,
    # aunque un inversor estricto podría preferir rellenar con 0. 
    score_columns = [c for c in df.columns if c.startswith('Score_')]
    df[score_columns] = df[score_columns].fillna(0.5)
    
    return df

def calculate_final_scores(df):
    # Categoría: Valuation (28.57%) - 60% PE, 40% PB
    df['Cat_Valuation'] = (df['Score_PE'] * 0.6) + (df['Score_PB'] * 0.4)
    
    # Categoría: Growth (28.57%) - 60% EPS, 40% Rev
    df['Cat_Growth'] = (df['Score_EPS_Growth'] * 0.6) + (df['Score_Rev_Growth'] * 0.4)
    
    # Categoría: Health (21.43%) - 60% Debt/Eq, 40% ROE
    df['Cat_Health'] = (df['Score_Debt_Equity'] * 0.6) + (df['Score_ROE'] * 0.4)
    
    # Categoría: Technical (21.43%) - 60% Momentum, 40% RSI
    df['Cat_Technical'] = (df['Score_Momentum'] * 0.6) + (df['Score_RSI'] * 0.4)
    
    # Puntuación Final
    df['Total_Score'] = (
        df['Cat_Valuation'] * WEIGHT_VALUATION +
        df['Cat_Growth'] * WEIGHT_GROWTH +
        df['Cat_Health'] * WEIGHT_HEALTH +
        df['Cat_Technical'] * WEIGHT_TECH
    )
    
    return df

def main():
    print("Iniciando Bot de Análisis del S&P 500...")
    tickers = get_sp500_tickers()
    
    # Para pruebas rápidas podemos limitar a 50 tickers.
    # Descomentar la siguiente línea si se quiere probar rápido
    # tickers = tickers[:50]
    
    print(f"Descargando datos para {len(tickers)} empresas. Esto tomará unos minutos...")
    
    results = []
    # Usar ThreadPool para acelerar descargas simultaneas de I/O (yfinance)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_data_for_ticker, t): t for t in tickers}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            if len(res) > 1: # tiene más que solo el Ticker
                results.append(res)
            if i % 50 == 0:
                print(f"Completado {i}/{len(tickers)}...")
                
    df = pd.DataFrame(results)
    
    print("Calculando puntuaciones...")
    df = normalize_metrics(df)
    df = calculate_final_scores(df)
    
    # Ordenar por puntuación total descendente
    df_sorted = df.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)
    
    top_10 = df_sorted.head(10)
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    report_filename = f"sp500_top10_{date_str}.md"
    
    # Generar reporte en Markdown
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"# Top 10 Valores del S&P 500 - {date_str}\n\n")
        f.write("Basado en el Multi-Factor Scoring Framework modificado (excluyendo datos de pago de macro/sentimiento).\n\n")
        
        # Crear tabla resumen
        cols_to_show = ['Ticker', 'Total_Score', 'Cat_Valuation', 'Cat_Growth', 'Cat_Health', 'Cat_Technical']
        f.write(top_10[cols_to_show].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        
        f.write("## Detalle de Métricas Brutas del Top 10\n\n")
        raw_cols = ['Ticker', 'PE', 'PB', 'EPS_Growth', 'Rev_Growth', 'Debt_Equity', 'ROE', 'Momentum', 'RSI']
        f.write(top_10[raw_cols].to_markdown(index=False, floatfmt=".2f"))
        
    print(f"¡Análisis completado! Reporte guardado en {report_filename}")

if __name__ == "__main__":
    main()
