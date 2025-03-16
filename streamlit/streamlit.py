import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


# Web App Title
st.markdown('''
# **The EDA App**

This is the **EDA App** created in Streamlit using the **pandas-profiling** library.
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file_features = st.sidebar.file_uploader("Upload your input CSV file of features", type=["csv"])
    uploaded_file_train = st.sidebar.file_uploader("Upload your input CSV file of matches", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Pandas Profiling Report
if uploaded_file_features is not None and  uploaded_file_train is not None:
    @st.cache_data
    def load_csv_features():
        csv = pd.read_csv(uploaded_file_features)
        return csv


    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file_train)
        return csv
    df = load_csv_features()
    df = df.drop(columns=['Unnamed: 0'])
    st.header('**Input DataFrame with features**')
    st.write(df.head(10))
    st.write('---')

    df_train = load_csv()
    df_train = df_train.drop(columns=['Unnamed: 0'])
    st.header('**Input DataFrame with target**')
    st.write(df_train.head(10))
    st.write('---')

    st.header('*Количество пропусков по фичам*')
    fig, ax = plt.subplots()
    ax.bar(df.columns, df.isna().sum())
    plt.xticks(rotation=90)
    plt.grid()
    st.pyplot(fig)

    st.header('*Гистограмма распределений топ стран-производителей*')
    fig, ax = plt.subplots()
    top_country = list(df['country'].value_counts().index)[:30]
    ax.hist(df[df['country'].isin(top_country)]['country'].astype(str), bins=150)
    plt.xticks(rotation=90)
    plt.grid()
    st.pyplot(fig)

    st.header('*Гистограмма распределений топ брендов*')
    fig, ax = plt.subplots()
    top_brand = list(df['brand'].value_counts().index)[:30]
    ax.hist(df[df['brand'].isin(top_brand)]['brand'].astype(str), bins=150)
    plt.xticks(rotation=90)
    plt.grid()
    st.pyplot(fig)

    st.header('*Количество уникальных значений*')
    fig, ax = plt.subplots()
    imp_features = ['brand', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'material', 'type']
    ax.bar(imp_features, df[imp_features].apply(lambda x: len(x.unique())))
    plt.xticks(rotation=90)
    plt.grid()
    st.pyplot(fig)

    st.header('*BoxPlot для количеств товара в упаковке*')
    fig, ax = plt.subplots()
    ax.boxplot(df['count_in_pack'])
    #plt.xticks(rotation=90)
    plt.grid()
    st.pyplot(fig)

    st.header('*Распределение количества таргетов*')
    fig, ax = plt.subplots()

    ax.bar(['0', '1'], [df_train[df_train['target'] == 0].shape[0], df_train[df_train['target'] == 1].shape[0]])
    plt.grid()
    st.pyplot(fig)


else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache_data
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')


def main():
    st.title("Система обучения и предсказания моделей")

    # Выбор действия: обучение или предсказание
    action = st.radio("Выберите действие", ["Предсказание", "Обучение модели"])

    if action == "Предсказание":
        upload_and_predict()
    elif action == "Обучение модели":
        upload_and_train()


def upload_and_predict():
    # Интерфейс для загрузки файла и выполнения предсказания
    uploaded_file = st.file_uploader("Загрузите файл (pickle) для предсказания", type=["pkl"])

    if uploaded_file is not None:
        if st.button("Отправить на предсказание"):
            # Отправка файла на сервер для предсказания
            files = {"file": ("uploaded_file.pkl", uploaded_file, "application/octet-stream")}
            url = "http://127.0.0.1:8000/predictions/predict"  # URL для предсказания

            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    st.success("Предсказание успешно выполнено!")
                    # Скачивание результатов
                    st.download_button(
                        label="Скачать результат",
                        data=response.content,
                        file_name="predictions_output.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Ошибка: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при отправке запроса: {e}")


def upload_and_train():
    # Интерфейс для обучения модели
    uploaded_file = st.file_uploader("Загрузите файл (pickle) для обучения модели", type=["pkl"])

    if uploaded_file is not None:
        # Настройка параметров для обучения
        model_name = st.text_input("Введите название модели")
        additional_training = st.checkbox("Дополнительное обучение", value=False)
        n_epochs = st.number_input("Количество эпох", min_value=1, max_value=100, value=10)
        batch_size = st.number_input("Размер батча", min_value=1, max_value=512, value=32)

        if st.button("Отправить на обучение"):
            # Отправка файла и параметров на сервер для обучения
            files = {"file": ("uploaded_file.pkl", uploaded_file, "application/octet-stream")}
            url = f"http://localhost:8000/train?model_name={model_name}&additional_training={additional_training}&n_epochs={n_epochs}&batch_size={batch_size}"

            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Модель успешно обучена: {result['message']}")
                    st.write(f"Средняя ошибка на тренировке: {result['avg_train_loss']}")
                    st.write(f"Средняя ошибка на тесте: {result['avg_test_loss']}")
                else:
                    st.error(f"Ошибка: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при отправке запроса: {e}")


if __name__ == "__main__":
    main()
