import streamlit as st
import requests

# Указываем адрес FastAPI сервера
url = "http://127.0.0.1:8000/predictions/predict"


def upload_file():
    # Загружаем файл с помощью Streamlit
    uploaded_file = st.file_uploader("Загрузите файл (pickle)", type=["pkl"])

    if uploaded_file is not None:
        # Отправляем файл на FastAPI сервер
        files = {"file": ("uploaded_file.pkl", uploaded_file, "application/octet-stream")}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                st.success("Файл успешно обработан, можете скачать файл с предсказаниями!")
                # Сохраняем полученный CSV файл
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



st.title("API для предсказаний")

upload_file()
