import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("Система виявлення фінансових махінацій")
st.markdown("---")

API_URL = "http://localhost:8000"

TRANSACTION_TYPES = ["ATM", "Online", "POS", "QR", "Transfer"]
MERCHANT_CATEGORIES = ["Clothing", "Electronics", "Food", "Gambling", "Grocery",
                       "Travel", "Utilities", "Other"]
COUNTRIES = ["UA", "US", "GB", "DE", "FR", "PL", "IT", "TR", "NG", "IN", "RU", "CN", "PK", "Other"]

# Очікувані колонки нового датасету (для пакетного завантаження)
REQUIRED_COLUMNS = [
    "user_id", "amount", "transaction_type", "merchant_category",
    "country", "hour", "device_risk_score", "ip_risk_score"
]


def call_api(endpoint: str, data: dict | list):
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Помилка API {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Не вдається підключитися до API. Переконайтеся, що сервер запущений: `python src/api.py`")
        return None
    except requests.exceptions.Timeout:
        st.error("Тайм-аут запиту до API")
        return None
    except Exception as e:
        st.error(f"Помилка: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Навігація")
page = st.sidebar.selectbox(
    "Оберіть сторінку",
    ["Перевірка транзакції", "Пакетна обробка", "Аналітика"]
)

# ---------------------------------------------------------------------------
# Сторінка 1: Перевірка транзакції
# ---------------------------------------------------------------------------
if page == "Перевірка транзакції":
    st.header("Перевірка окремої транзакції")

    # Статус API
    st.subheader("Статус API")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "healthy":
            st.success("API сервер працює")
        else:
            st.error(f"API нездоровий: {health}")
    except Exception:
        st.error("API недоступний. Запустіть: `python src/api.py`")

    st.subheader("Введіть дані транзакції")
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.text_input("ID користувача", value="363")
        amount = st.number_input("Сума транзакції", min_value=0.01, value=100.0, step=0.01)
        transaction_type = st.selectbox("Тип транзакції", TRANSACTION_TYPES)
        merchant_category = st.selectbox("Категорія мерчанта", MERCHANT_CATEGORIES)

    with col2:
        country = st.selectbox("Країна", COUNTRIES)
        hour = st.slider("Година доби (0–23)", min_value=0, max_value=23, value=datetime.now().hour)
        device_risk_score = st.slider("Ризик-скор пристрою", 0.0, 1.0, value=0.1, step=0.01)
        ip_risk_score = st.slider("Ризик-скор IP", 0.0, 1.0, value=0.1, step=0.01)

    transaction_data = {
        "user_id": user_id,
        "amount": amount,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "country": country,
        "hour": hour,
        "device_risk_score": device_risk_score,
        "ip_risk_score": ip_risk_score
    }

    st.subheader("Підсумок транзакції")
    st.json(transaction_data)

    if st.button("Перевірити транзакцію", type="primary"):
        with st.spinner("Аналізуємо транзакцію..."):
            result = call_api("/score", transaction_data)

        if result:
            col1, col2, col3 = st.columns(3)

            with col1:
                fraud_prob = result.get("fraud_probability", 0.0)
                st.metric("Ймовірність махінації", f"{fraud_prob:.2%}")

            with col2:
                decision = result.get("decision", "UNKNOWN")
                color_map = {"ALLOW": "green", "REVIEW": "orange", "BLOCK": "red", "UNKNOWN": "gray"}
                color = color_map.get(decision, "gray")
                st.markdown(f"**Рішення:** :{color}[{decision}]")

            with col3:
                risk_level = result.get("risk_level", "UNKNOWN")
                st.markdown(f"**Рівень ризику:** {risk_level}")

            st.subheader("Пояснення")
            st.info(result.get("explanation", "Пояснення недоступне"))

            st.subheader("Найважливіші ознаки")
            top_features = result.get("top_features", {})
            if top_features:
                features_df = pd.DataFrame(
                    list(top_features.items()),
                    columns=["Ознака", "Внесок (SHAP)"]
                )
                fig = px.bar(
                    features_df, x="Внесок (SHAP)", y="Ознака",
                    orientation="h", title="Внесок ознак у рішення (SHAP)"
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Сторінка 2: Пакетна обробка
# ---------------------------------------------------------------------------
elif page == "Пакетна обробка":
    st.header("Пакетна обробка транзакцій")

    st.info(
        f"**Формат CSV файлу.** Обов'язкові колонки: `{'`, `'.join(REQUIRED_COLUMNS)}`\n\n"
        f"Колонки `transaction_id` та `is_fraud` допускаються, але ігноруються."
    )

    # Шаблон для завантаження
    if st.button("Завантажити шаблон CSV"):
        sample = {
            "user_id": ["363", "692", "445"],
            "amount": [4922.59, 48.02, 80.53],
            "transaction_type": ["ATM", "QR", "POS"],
            "merchant_category": ["Travel", "Food", "Clothing"],
            "country": ["TR", "US", "TR"],
            "hour": [12, 21, 23],
            "device_risk_score": [0.99, 0.17, 0.12],
            "ip_risk_score": [0.95, 0.22, 0.16]
        }
        csv_template = pd.DataFrame(sample).to_csv(index=False)
        st.download_button(
            label="Завантажити шаблон",
            data=csv_template,
            file_name="transaction_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Завантажте CSV файл", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

            if missing:
                st.error(f"Відсутні колонки: {', '.join(missing)}")
            else:
                st.success(f"Файл завантажено: {len(df)} транзакцій")
                st.dataframe(df.head(10))

                if st.button("Обробити всі транзакції", type="primary"):
                    transactions = df[REQUIRED_COLUMNS].to_dict("records")

                    with st.spinner(f"Обробляємо {len(transactions)} транзакцій..."):
                        result = call_api("/batch_score", transactions)

                    if result:
                        results_list = result.get("results", [])
                        if not results_list:
                            st.warning("Результати порожні")
                        else:
                            results_df = pd.DataFrame(results_list)

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Всього", len(results_df))
                            with col2:
                                block = (results_df["decision"] == "BLOCK").sum()
                                st.metric("Заблоковано", block)
                            with col3:
                                review = (results_df["decision"] == "REVIEW").sum()
                                st.metric("На перевірку", review)
                            with col4:
                                allow = (results_df["decision"] == "ALLOW").sum()
                                st.metric("Дозволено", allow)

                            if "decision" in results_df.columns:
                                counts = results_df["decision"].value_counts()
                                fig = px.pie(values=counts.values, names=counts.index,
                                             title="Розподіл рішень")
                                st.plotly_chart(fig, use_container_width=True)

                            st.dataframe(results_df)

                            st.download_button(
                                label="Завантажити результати",
                                data=results_df.to_csv(index=False),
                                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

        except Exception as e:
            st.error(f"Помилка читання файлу: {e}")

# ---------------------------------------------------------------------------
# Сторінка 3: Аналітика
# ---------------------------------------------------------------------------
elif page == "Аналітика":
    st.header("Аналітика системи")

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "healthy":
            st.success("API працює нормально")
        else:
            st.warning(f"Стан API: {health.get('status')}")
    except Exception:
        st.error("API недоступний")

    st.subheader("Демонстраційна статистика")
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    fraud_rates = np.random.default_rng(42).uniform(0.01, 0.05, 30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=fraud_rates, mode="lines+markers",
                             name="Частота махінацій"))
    fig.update_layout(
        title="Динаміка виявлення махінацій",
        xaxis_title="Дата",
        yaxis_title="Частота махінацій"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("**Fraud Detection System v2.0** | Дипломна робота")
