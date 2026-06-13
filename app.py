import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# 페이지 설정
st.set_page_config(page_title="Used Car Price Prediction", layout="wide")

# 헤더
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚗 Used Car Price Prediction Dashboard</h1>", unsafe_allow_html=True)

# 데이터 불러오기
DATA_PATH = "CarPrice_Assignment.csv"   # Kaggle에서 받은 CSV 파일
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("❌ car_data.csv 파일을 찾을 수 없습니다. Kaggle에서 다운로드 후 같은 폴더에 넣어주세요.")
    st.stop()

# 탭 구조
tab1, tab2, tab3 = st.tabs(["📊 데이터 탐색", "📈 모델 비교", "🔍 가격 예측"])

# -------------------- TAB 1: 데이터 탐색 --------------------
with tab1:
    st.subheader("데이터 미리보기")
    st.write(data.head())

    st.subheader("데이터 통계 요약")
    st.write(data.describe())

    st.subheader("연료 종류별 평균 판매 가격")
    if "Fuel_Type" in data.columns:
        fuel_price = data.groupby("Fuel_Type")["Selling_Price"].mean()
        st.bar_chart(fuel_price)

    st.subheader("연식 vs 판매 가격")
    if "Year" in data.columns:
        st.line_chart(data.groupby("Year")["Selling_Price"].mean())

# -------------------- TAB 2: 모델 비교 --------------------
with tab2:
    st.subheader("모델 성능 비교")

    # 특징과 라벨 분리
    X = data.drop("Selling_Price", axis=1)
    y = data["Selling_Price"]

    # 범주형 변수 처리
    X = pd.get_dummies(X, drop_first=True)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 모델 리스트
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {"RMSE": rmse, "R2": r2}

        st.markdown(f"### {name}")
        st.write(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # 성능 비교 시각화
    st.subheader("📌 모델별 성능 비교")
    results_df = pd.DataFrame(results).T
    st.bar_chart(results_df["R2"])

# -------------------- TAB 3: 가격 예측 --------------------
with tab3:
    st.subheader("새로운 차량 가격 예측")

    input_data = {}
    for col in X.columns:
        val = st.number_input(f"{col} 값 입력", value=float(X[col].mean()))
        input_data[col] = val

    input_df = pd.DataFrame([input_data])

    model_choice = st.selectbox("모델 선택", list(models.keys()))
    chosen_model = models[model_choice]
    chosen_model.fit(X_train, y_train)
    prediction = chosen_model.predict(input_df)[0]

    st.markdown("<h3 style='color: green;'>🚗 예상 판매 가격</h3>", unsafe_allow_html=True)
    st.success(f"{prediction:.2f} Lakhs")
