# Streamlit интерфейс
import streamlit as st
st.set_page_config(page_title="Credit Coach BCC", page_icon="🏦")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Кастомизация темы (белый фон, зелёные акценты БЦК)
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .stButton>button {
        background-color: #00A859;
        color: white;
        border-radius: 5px;
    }
    .stMetric {
        border: 2px solid #00A859;
        border-radius: 5px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #00A859;
    }
    </style>
""", unsafe_allow_html=True)

# Генерация синтетической базы данных
np.random.seed(42)
n_samples = 1000
data = {
    'income': np.random.normal(300000, 50000, n_samples),
    'expenses': np.random.normal(180000, 30000, n_samples),
    'late_payments': np.random.randint(0, 5, n_samples),
    'account_balance': np.random.normal(50000, 10000, n_samples),
    'fines': np.random.randint(0, 3, n_samples),
    'tax_debt': np.random.normal(5000, 2000, n_samples),
    'active_loans': np.random.randint(0, 5, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples)
}
df = pd.DataFrame(data)
X = df[['income', 'expenses', 'late_payments', 'account_balance', 'fines', 'tax_debt', 'active_loans']]
y = df['credit_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Функция-заглушка для API eGov.kz
def check_egov_data(iin):
    fines = np.random.randint(0, 3)
    tax_debt = np.random.normal(5000, 2000)
    return fines, tax_debt

# Функция для оценки одобрения с учётом свободного баланса
def approval_probability(credit_score, monthly_payment, income, expenses, active_loans, loan_payments):
    free_balance = income - expenses - loan_payments  # Свободный баланс
    if credit_score < 600 or monthly_payment > free_balance * 0.5 or active_loans > 3:
        return False, "Кредит не одобрен: низкий рейтинг, недостаточный свободный баланс или слишком много кредитов."
    return True, "Кредит, скорее всего, будет одобрен."

# Функция рекомендаций (более детализированная)
def generate_recommendations(client_data, loan_amount, loan_term_months, iin, loan_payments):
    # client_data: [income, expenses, late_payments, account_balance, fines, tax_debt, active_loans]
    # loan_payments: сумма платежей по текущим кредитам (₸)
    
    fines, tax_debt = check_egov_data(iin)
    client_data[4] = fines
    client_data[5] = tax_debt
    current_score = model.predict([client_data])[0]
    monthly_payment = loan_amount / loan_term_months
    free_balance = client_data[0] - client_data[1] - loan_payments  # Свободный баланс
    recommendations = []
    loan_suggestions = []

    # Детализированные рекомендации
    if client_data[2] > 0:
        recommendations.append(
            f"У вас {client_data[2]} просроченных платежей. Погасите их в первую очередь, так как они сильно снижают ваш кредитный рейтинг. "
            "Проверьте задолженности в мобильном приложении БЦК или через eGov.kz и оплатите их онлайн."
        )
    
    if client_data[4] > 0:
        recommendations.append(
            f"Обнаружено {client_data[4]} неоплаченных штрафов (например, за нарушение ПДД). "
            "Зайдите на eGov.kz, раздел 'Штрафы', оплатите их через карту и обновите данные через 3–5 дней."
        )
    
    if client_data[5] > 0:
        recommendations.append(
            f"У вас долг по налогам: {client_data[5]:.2f} ₸. Это может быть причиной отказа в кредите. "
            "Перейдите на eGov.kz, раздел 'Налоги', погасите задолженность и дождитесь обновления данных (обычно 5–7 дней)."
        )
    
    if client_data[6] > 2:
        recommendations.append(
            f"У вас {client_data[6]} активных кредитов, что создаёт высокую долговую нагрузку. "
            "Рассмотрите возможность закрыть хотя бы один кредит (например, с наименьшей суммой). "
            "Это можно сделать через приложение БЦК или в отделении."
        )
    
    if client_data[1] > client_data[0] * 0.5:
        recommendations.append(
            "Ваши расходы составляют более 50% дохода, что снижает вашу финансовую стабильность в глазах банка. "
            "Попробуйте сократить необязательные траты (например, подписки или частые покупки) на 10–20% в течение 1–2 месяцев."
        )
    
    if free_balance < monthly_payment * 2:
        recommendations.append(
            f"Ваш свободный баланс после текущих расходов и платежей по кредитам: {free_balance:.2f} ₸. "
            "Новый кредитный платёж ({monthly_payment:.2f} ₸) превышает 50% свободного баланса, что снижает шансы на одобрение."
        )

    # Проверка одобрения
    approved, approval_reason = approval_probability(current_score, monthly_payment, client_data[0], client_data[1], client_data[6], loan_payments)
    if not approved:
        recommendations.append(approval_reason)
        new_term = int(loan_amount / (free_balance * 0.5)) + 1
        new_payment = loan_amount / new_term
        loan_suggestions.append(
            f"Увеличьте срок кредита до {new_term} месяцев — это снизит ежемесячный платёж до {new_payment:.2f} ₸, "
            "что уложится в ваш свободный баланс и повысит шансы на одобрение."
        )
        new_amount = free_balance * 0.5 * loan_term_months
        loan_suggestions.append(
            f"Или уменьшите сумму кредита до {new_amount:.2f} ₸ для текущего срока ({loan_term_months} месяцев), "
            "чтобы платёж был в пределах 50% вашего свободного баланса."
        )
        if current_score < 600:
            loan_suggestions.append(
                "Ваш кредитный рейтинг ниже 600. Рассмотрите привлечение созаёмщика или поручителя с хорошей кредитной историей. "
                "Это можно оформить в отделении БЦК, предоставив документы созаёмщика."
            )
        loan_suggestions.append(
            "Подайте заявку повторно через 2 недели — данные в кредитном бюро обновятся, и ваши шансы на одобрение вырастут. "
            "Следите за обновлениями в приложении БЦК."
        )
    
    if not recommendations:
        recommendations.append("Ваши финансы в отличном состоянии! Продолжайте следить за своим рейтингом через приложение БЦК.")

    return recommendations, loan_suggestions, current_score, approved

# Добавление логотипа БЦК
st.image("images/bcclog.png", width=600) 

st.title("Credit Coach BCC")
st.subheader("Проверьте ваш кредитный рейтинг и получите персональные рекомендации") 

# Форма для ввода данных
with st.form("client_form"):
    iin = st.text_input("Введите ваш ИИН (12 цифр)", max_chars=12)
    income = st.number_input("Месячный доход (₸)", min_value=0, value=250000)
    expenses = st.number_input("Месячные расходы (₸)", min_value=0, value=150000)
    late_payments = st.number_input("Количество просрочек", min_value=0, value=1)
    active_loans = st.number_input("Количество активных кредитов", min_value=0, value=3)
    loan_payments = st.number_input("Сумма платежей по текущим кредитам (₸/месяц)", min_value=0, value=50000)
    loan_amount = st.number_input("Сумма кредита (₸)", min_value=0, value=3000000)
    loan_term_months = st.number_input("Срок кредита (месяцы)", min_value=1, value=6)
    submitted = st.form_submit_button("Проверить")

if submitted:
    client_data = [income, expenses, late_payments, 50000, 0, 0, active_loans]  # account_balance не используется
    recommendations, loan_suggestions, current_score, approved = generate_recommendations(
        client_data, loan_amount, loan_term_months, iin, loan_payments
    )

    # Вывод результатов
    st.header("Результаты")
    st.metric("Ваш кредитный рейтинг", f"{current_score:.2f}")
    st.metric("Статус заявки", "Одобрено" if approved else "Отказано")

    st.subheader("Рекомендации для улучшения кредитного рейтинга")
    for rec in recommendations:
        st.write(f"- {rec}")

    if loan_suggestions:
        st.subheader("Предложения по кредиту")
        for suggestion in loan_suggestions:
            st.write(f"- {suggestion}")
    else:
        st.success("Ваш кредит, скорее всего, одобрят на текущих условиях!")

    # Улучшенный график
    st.subheader("Какие факторы влияют на Ваш рейтинг?")
    factors = ['Доходы', 'Расходы', 'Просрочки', 'Штрафы', 'Налоги', 'Кредиты']
    importance = model.feature_importances_[[0, 1, 2, 4, 5, 6]]  # Исключаем account_balance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importance, y=factors, palette=['#00A859' if i < 3 else '#66C2A5' for i in range(len(factors))])
    ax.set_xlabel("Влияние на рейтинг", fontsize=12)
    ax.set_title("Какие факторы влияют на Ваш рейтинг?", fontsize=14, pad=20)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    for i, v in enumerate(importance):
        ax.text(v + 0.01, i, f"{v:.2%}", va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)