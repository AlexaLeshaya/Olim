# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Убедимся, что openpyxl установлен
try:
    import openpyxl
except ImportError:
    raise ImportError("Необходимо установить библиотеку 'openpyxl'. Выполните команду 'pip install openpyxl'.")

# Функция для обработки данных
def process_data():
    # Загрузка данных из файла в репозитории
    file_path = "Данные_Экономическая_Безопасность_Таджикистан_Полные.xlsx"
    data_customers = pd.read_excel(file_path, sheet_name="Клиенты")
    data_transactions = pd.read_excel(file_path, sheet_name="Финансовые транзакции")
    data_usage = pd.read_excel(file_path, sheet_name="Потребление ресурсов")

    # Очистка данных
    cleaned_customers = data_customers.dropna()
    cleaned_transactions = data_transactions.dropna()
    cleaned_usage = data_usage.dropna()

    # Объединение данных
    merged_data = pd.merge(cleaned_transactions, cleaned_customers, on="ID клиента", how="inner")
    merged_data = pd.merge(merged_data, cleaned_usage, on="ID клиента", how="inner")

    return merged_data

# Функция для расчета Z-оценки
def calculate_z_score(column):
    return (column - column.mean()) / column.std()

# Функция для визуализации данных
def visualize_data(merged_data):
    # Распределение задолженностей
    st.subheader("Распределение задолженностей")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(merged_data["Общая задолженность (сомони)"], bins=30, kde=True, ax=ax)
    ax.set_title("Распределение задолженностей")
    ax.set_xlabel("Задолженность (сомони)")
    ax.set_ylabel("Частота")
    st.pyplot(fig)

    # Распределение потребления ресурсов
    st.subheader("Распределение потребления ресурсов")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Тип ресурса", y="Объем потребления", data=merged_data, ax=ax)
    ax.set_title("Распределение потребления ресурсов")
    ax.set_xlabel("Тип ресурса")
    ax.set_ylabel("Объем потребления")
    st.pyplot(fig)

    # Тепловая карта корреляции
    st.subheader("Тепловая карта корреляции")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = merged_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Тепловая карта корреляции")
    st.pyplot(fig)

# Функция для обработки пользовательских данных и расчета потерь
def calculate_risks(merged_data):
    # Выявление задолженностей
    threshold_debt = st.sidebar.slider("Порог задолженности (сомони)", 100, 500, 200)
    merged_data["Задолженность"] = merged_data["Общая задолженность (сомони)"] > threshold_debt

    # Выявление аномалий в потреблении
    merged_data["Z-оценка потребления"] = calculate_z_score(merged_data["Объем потребления"])
    merged_data["Аномалия потребления"] = merged_data["Z-оценка потребления"].abs() > 3

    # Расчет потерь
    merged_data["Риск"] = np.where(merged_data["Задолженность"] | merged_data["Аномалия потребления"], 1, 0)
    def calculate_loss(row):
        return row["Риск"] * (row["Сумма (сомони)"] + row["Общая задолженность (сомони)"])

    merged_data["Потери"] = merged_data.apply(calculate_loss, axis=1)
    total_loss = merged_data["Потери"].sum()

    st.subheader("Общие потери")
    st.write(f"Общие потери: {total_loss:.2f} сомони")

    return merged_data

# Основная функция Streamlit
def main():
    st.title("Модель мониторинга и реагирования на угрозы экономической безопасности ЖКХ")

    # Загрузка данных из репозитория
    merged_data = process_data()
    st.success("Данные успешно загружены и обработаны")

    # Визуализация данных
    visualize_data(merged_data)

    # Расчёт рисков и потерь
    st.sidebar.header("Настройки анализа")
    result_data = calculate_risks(merged_data)

    # Вывод обработанных данных
    st.subheader("Обработанные данные")
    st.dataframe(result_data.head())

if __name__ == "__main__":
    main()
