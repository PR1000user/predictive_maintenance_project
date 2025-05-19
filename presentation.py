import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
<div style="font-size:50px;white-space: pre-wrap;text-align: center;">

## ⚙️ Прогнозирование отказов оборудования

</div>
---
## 🔍 Введение
<div style="font-size:20px; line-height:1; white-space: nowrap;text-align: left;">

В проектной работе создано приложение для прогнозирования отказов оборудования.

- Используется датасет **AI4I 2020 Predictive Maintenance Dataset** от Калифорнийского университета в Ирвайне

- Датасет содержит более 10 000 записей о работе промышленных станков. 

- Цель: предсказать **отказ оборудования** (Target = 1) или его отсутствие (Target = 0).  

- Задача **бинарной классификации**.
---
## 🧾 Обзор признаков
<div style="font-size:20px; line-height:1; white-space: nowrap;text-align: left;">

📌 Идентификаторы  

- `UID` — уникальный идентификатор записи  
- `Product ID` — ID изделия  

🏭 Тип оборудования  

- `Type` — категориальный признак: тип станка (`L`, `M`, `H`)  

🌡️ Параметры процесса  

- `Air temperature [K]` — температура окружающей среды  
- `Process temperature [K]` — температура процесса  
- `Rotational speed [rpm]` — скорость вращения  
- `Torque [Nm]` — крутящий момент  
- `Tool wear [min]` — степень износа инструмента  

⚠️ Целевые переменные  

- `Machine failure` — общий признак отказа (цель)  
- `TWF` — недостаток охлаждения  
- `HDF` — перегрев  
- `PWF` — перегрузка  
- `OSF` — отказ датчика  
- `RNF` — случайный отказ  
</div>
---
## 🛠 Этапы работы
<div style="font-size:20px; line-height:1; white-space: nowrap;text-align: left;">

Приведем основные этапы работы:

𝟏 **Загрузка данных.**

- Проверка на пропуски.    

𝟐 **Предобработка данных.**

- Кодирование категорий, масштабирование чисел.    

𝟑 **Обучение модели**

- Алгоритмы: Logistic Regression, Random Forest, XGBoost, Support Vector Machine.  

𝟒 **Оценка модели.**   

- Метрики: Accuracy, Recall, Precision, F1-score, Support.  

𝟓 **Визуализация результатов**  

- Графики ROC-AUC, confusion matrix, численные значения метрик.
</div>
---
## 🌐 Streamlit-приложение

<div style="font-size:20px; line-height:1; white-space: nowrap;text-align: left;">

Описание разработанного приложения:

𝟏 Основная страница: 

- анализ параметров различными алгоритмами
- визуализация
- предсказание по пользовательским данным.

𝟐 Страница с презентацией: 

- описание проекта
</div>
---
## 🏁 Заключение
<div style="font-size:20px; line-height:1; white-space: nowrap;text-align: left;">

✅ Подведение итогов

- Построена модель предиктивного обслуживания.  
- Высокая точность и интерпретируемость предсказанного результата.  
- Возможность интеграции в производственные процессы.

🚀 Возможные улучшения  

- **Многоцелевое прогнозирование**: предсказание не только общего отказа, но и его типа (`TWF`, `HDF`, и т.д.).  
- Использование **LSTM/GRU** для учёта временной динамики.  
- Применение **Explainable AI (SHAP, LIME)** для объяснения решений модели.  
- Учет **времени до отказа (Remaining Useful Life)**.  
- Интеграция с **промышленными IoT-системами**.
</div>
---
## 🤝 Спасибо за внимание!
"""

    # Настройки презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=700)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

# Запуск страницы презентации
if __name__ == "__main__":
    presentation_page()
