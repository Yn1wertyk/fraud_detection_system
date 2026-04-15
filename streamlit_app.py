import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Конфігурація сторінки
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Система виявлення фінансових махінацій")
st.markdown("---")

# API endpoint
API_URL = "http://localhost:8000"

def call_api(endpoint, data):
    """
    Виклик API
    """
    try:
        st.write(f"🔄 Відправляємо запит до: {API_URL}{endpoint}")
        st.write(f"📤 Дані запиту: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
        
        st.write(f"📥 Статус відповіді: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"✅ Отримано відповідь: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return result
        else:
            st.error(f"❌ Помилка API: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Не вдається підключитися до API сервера. Переконайтеся, що сервер запущений командою: `python src/api.py`")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Тайм-аут запиту до API")
        return None
    except Exception as e:
        st.error(f"❌ Помилка API: {str(e)}")
        return None

# Sidebar для навігації
st.sidebar.title("Навігація")
page = st.sidebar.selectbox("Оберіть сторінку", 
                           ["Перевірка транзакції", "Пакетна обробка", "Аналітика"])

if page == "Перевірка транзакції":
    st.header("Перевірка окремої транзакції")
    
    st.subheader("Статус API")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("status") == "healthy":
                st.success("✅ API сервер працює")
            else:
                st.error(f"❌ API нездоровий: {health_data}")
        else:
            st.error(f"❌ API повернув статус: {health_response.status_code}")
    except:
        st.error("❌ API сервер недоступний. Запустіть його командою: `python src/api.py`")
        st.info("💡 Для запуску API відкрийте новий термінал і виконайте: `cd fraud-detection-system && python src/api.py`")
    
    st.subheader("Введіть дані транзакції")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input("ID користувача", value="user_001", help="Унікальний ідентифікатор користувача")
        amount = st.number_input("Сума транзакції", min_value=0.01, value=100.0, step=0.01)
        currency = st.selectbox("Валюта", ["UAH", "USD", "EUR", "GBP"], index=0)
        merchant = st.selectbox("Тип мерчанта", [
            "grocery_store", "coffee_shop", "pharmacy", "gas_station", "restaurant", "bookstore",
            "electronics_store", "jewelry_shop", "luxury_goods", "travel_agency", "gift_cards",
            "crypto_exchange", "online_casino", "forex_trading", "adult_content", "vpn_service"
        ], index=0)
    
    with col2:
        device_id = st.text_input("ID пристрою", value="device_mobile_001", help="Ідентифікатор пристрою користувача")
        country = st.selectbox("Країна", ["UA", "US", "GB", "DE", "FR", "PL", "IT", "RU", "Unknown", "TOR", "VPN"], index=0)
        tx_date = st.date_input("Дата транзакції", value=datetime.now().date())
        tx_time = st.time_input("Час транзакції", value=datetime.now().time())
    
    # Combine date and time
    tx_datetime = datetime.combine(tx_date, tx_time)
    tx_time_str = tx_datetime.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Display transaction summary
    st.subheader("Підсумок транзакції")
    transaction_summary = {
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
        "merchant": merchant,
        "device_id": device_id,
        "country": country,
        "tx_time": tx_time_str
    }
    
    st.json(transaction_summary)
    
    if st.button("Перевірити транзакцію", type="primary"):
        with st.spinner("Аналізуємо транзакцію..."):
            result = call_api("/score", transaction_summary)
        
        if result:
            try:
                # Відображення результатів
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fraud_prob = result.get("fraud_probability", 0.0)
                    st.metric("Ймовірність махінації", f"{fraud_prob:20%}")
                
                with col2:
                    decision = result.get("decision", "UNKNOWN")
                    color_map = {"ALLOW": "green", "REVIEW": "orange", "BLOCK": "red", "UNKNOWN": "gray"}
                    color = color_map.get(decision, "gray")
                    st.markdown(f"**Рішення:** :{color}[{decision}]")
                
                with col3:
                    risk_level = result.get("risk_level", "UNKNOWN")
                    st.markdown(f"**Рівень ризику:** {risk_level}")
                
                # Пояснення
                st.subheader("Пояснення")
                explanation = result.get("explanation", "Пояснення недоступне")
                st.info(explanation)
                
                # Топ ознак
                st.subheader("Найважливіші ознаки")
                top_features = result.get("top_features", {})
                if top_features:
                    features_df = pd.DataFrame(
                        list(top_features.items()),
                        columns=["Ознака", "Внесок"]
                    )
                    
                    fig = px.bar(features_df, x="Внесок", y="Ознака", 
                                orientation="h", title="Внесок ознак у рішення")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Інформація про ознаки недоступна")
                    
            except Exception as e:
                st.error(f"Помилка обробки відповіді API: {str(e)}")
                st.json(result)  # Show raw response for debugging
        else:
            st.error("Не вдалося отримати відповідь від API. Переконайтеся, що API сервер запущений на http://localhost:8000")

elif page == "Пакетна обробка":
    st.header("Пакетна обробка транзакцій")
    
    st.subheader("Завантаження файлу з транзакціями")
    
    # Show required format
    st.info("""
    **Формат CSV файлу:**
    Файл повинен містити наступні колонки: `user_id`, `amount`, `currency`, `merchant`, `device_id`, `country`, `tx_time`
    
    **Приклад формату дати:** 2024-09-06T14:30:00
    """)
    
    # Sample template download
    if st.button("📥 Завантажити шаблон CSV"):
        sample_data = {
            'user_id': ['user_001', 'user_002', 'user_003'],
            'amount': [100.0, 500.0, 1200.0],
            'currency': ['UAH', 'USD', 'EUR'],
            'merchant': ['coffee_shop', 'online_casino', 'electronics_store'],
            'device_id': ['device_mobile_001', 'device_unknown_999', 'device_laptop_003'],
            'country': ['UA', 'RU', 'DE'],
            'tx_time': ['2024-09-06T09:15:00', '2024-09-06T03:22:00', '2024-09-06T14:30:00']
        }
        sample_df = pd.DataFrame(sample_data)
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Завантажити шаблон",
            data=csv,
            file_name="transaction_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Завантажте CSV файл з транзакціями", 
                                   type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['user_id', 'amount', 'currency', 'merchant', 'device_id', 'country', 'tx_time']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Відсутні обов'язкові колонки: {', '.join(missing_columns)}")
                st.info("Переконайтеся, що ваш файл містить всі необхідні колонки.")
            else:
                st.success(f"✅ Файл завантажено успішно! Знайдено {len(df)} транзакцій")
                
                st.subheader("Попередній перегляд даних")
                st.dataframe(df.head(10))
                
                if st.button("Обробити всі транзакції", type="primary"):
                    # Конвертуємо DataFrame в список транзакцій
                    transactions = df.to_dict('records')
                    
                    with st.spinner(f"Обробляємо {len(transactions)} транзакцій..."):
                        result = call_api("/batch_score", transactions)
                    
                    if result:
                        try:
                            results_list = result.get("results", [])
                            if not results_list:
                                st.warning("Результати обробки порожні")
                            else:
                                results_df = pd.DataFrame(results_list)
                                
                                # Статистика
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    total_count = len(results_df)
                                    st.metric("Всього транзакцій", total_count)
                                
                                with col2:
                                    fraud_count = len(results_df[results_df.get("decision", "") == "BLOCK"])
                                    st.metric("Заблоковано", fraud_count)
                                
                                with col3:
                                    review_count = len(results_df[results_df.get("decision", "") == "REVIEW"])
                                    st.metric("На перевірку", review_count)
                                
                                with col4:
                                    allow_count = len(results_df[results_df.get("decision", "") == "ALLOW"])
                                    st.metric("Дозволено", allow_count)
                                
                                # Розподіл рішень
                                if "decision" in results_df.columns:
                                    decision_counts = results_df["decision"].value_counts()
                                    fig = px.pie(values=decision_counts.values, names=decision_counts.index,
                                               title="Розподіл рішень")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Таблиця результатів з можливістю завантаження
                                st.subheader("Результати")
                                st.dataframe(results_df)
                                
                                # Download results
                                csv_results = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Завантажити результати",
                                    data=csv_results,
                                    file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                        except Exception as e:
                            st.error(f"Помилка обробки пакетних результатів: {str(e)}")
                            st.json(result)  # Show raw response for debugging
                    else:
                        st.error("Не вдалося отримати відповідь від API. Переконайтеся, що API сервер запущений на http://localhost:8000")
        
        except Exception as e:
            st.error(f"Помилка читання файлу: {str(e)}")
            st.info("Переконайтеся, що файл має правильний CSV формат з комами як роздільниками.")

elif page == "Аналітика":
    st.header("Аналітика системи")
    
    # Перевірка здоров'я API
    try:
        health = requests.get(f"{API_URL}/health").json()
        if health["status"] == "healthy":
            st.success("✅ API працює нормально")
        else:
            st.error("❌ Проблеми з API")
    except:
        st.error("❌ API недоступний")
    
    # Демонстраційна аналітика
    st.subheader("Демонстраційна статистика")
    
    # Генеруємо фейкові дані для демонстрації
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    fraud_rates = np.random.uniform(0.01, 0.05, 30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=fraud_rates, mode='lines+markers',
                           name='Частота махінацій'))
    fig.update_layout(title="Динаміка виявлення махінацій",
                     xaxis_title="Дата", yaxis_title="Частота махінацій")
    st.plotly_chart(fig, use_container_width=True)

# Футер
st.markdown("---")
st.markdown("**Fraud Detection System v1.0** | Розроблено для дипломної роботи")
