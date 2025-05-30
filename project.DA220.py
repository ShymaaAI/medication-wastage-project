import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json
from datetime import datetime

# --- تحميل وتنظيف البيانات ---
DATA_PATH = "database/Untitled2.xlsx"

@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.replace('[^\w_]', '', regex=True)
    df.fillna("غير محدد", inplace=True)

    response_mapping = {
        "لا أوافق بشدة": 1,
        "لا أوافق": 2,
        "محايد": 3,
        "أوافق": 4,
        "أوافق بشدة": 5
    }

    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).intersection(set(response_mapping.keys())):
                df[col] = df[col].map(response_mapping).fillna(df[col])

    feature_cols = df.columns[6:].tolist()

    threshold = len(feature_cols) * 3.0

    X = df[feature_cols]
    df["Awareness_Level"] = X.sum(axis=1).apply(lambda x: 1 if x >= threshold else 0)
    y = df["Awareness_Level"]

    return df, X, y, feature_cols

df, X, y, feature_cols = load_and_prepare_data()

# --- تدريب النموذج لو مش موجود ---
MODEL_PATH = "streamlit_app/medication_wastage_model.pkl"

def train_and_save_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_and_save_model()

# --- إعداد أسئلة واجهة المستخدم ---
display_questions = [
    "أحرص على استكمال الجرعة الكاملة من الدواء حسب وصفة الطبيب",
    "أشارك معلومات عن الأدوية مع أشخاص آخرين عند الحاجة",
    "أشتري أدوية بدون وصفة طبية",
    "أقرأ النشرة المرفقة مع الدواء قبل استخدامه",
    "أخبر الطبيب بأن لدي نفس الدواء مسبقًا",
    "أشعر بالسعادة عندما يصف الطبيب أدوية أكثر",
    "أفضل الدواء الأجنبي على المحلي",
    "أحتفظ بالأدوية المتبقية لاستخدامها لاحقًا",
    "أستشير الصيدلي بخصوص الأدوية القديمة",
    "التخلص الخاطئ من الأدوية يضر البيئة"
]

likert_options = ["لا أوافق بشدة", "لا أوافق", "محايد", "أوافق", "أوافق بشدة"]
mapping = {"لا أوافق بشدة":1, "لا أوافق":2, "محايد":3, "أوافق":4, "أوافق بشدة":5}

# --- واجهة Streamlit ---
st.set_page_config(page_title="Jordanians' Awareness of Medication Wastage", layout="centered")
st.title("💊 تقييم وعي الأردنيين بهدر الأدوية")

name = st.text_input("الاسم الكامل:")
student_id = st.text_input("الرقم الجامعي:")

st.markdown("يرجى اختيار إجاباتك بعناية من القائمة أدناه ثم اضغط على زر التنبؤ.")

user_responses = []
with st.form("form"):
    for q in display_questions:
        answer = st.selectbox(q, likert_options)
        user_responses.append(mapping[answer])
    submitted = st.form_submit_button("تنبؤ مستوى الوعي")

# --- تخزين الردود في ملف JSON ---
def load_data():
    file_path = "database/responses_data.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

def save_data(data):
    file_path = "database/responses_data.json"
    existing_data = load_data()
    for record in existing_data:
        if record["student_id"] == data["student_id"]:
            record.update(data)
            break
    else:
        existing_data.append(data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# --- التنبؤ وإظهار النتائج ---
if submitted:
    if not name.strip() or not student_id.strip():
        st.error("الرجاء إدخال الاسم الكامل والرقم الجامعي.")
    else:
        avg_score = sum(user_responses) / len(user_responses)
        if avg_score >= 3.5:
            st.success("🎉 لديك وعي مرتفع بهدر الأدوية! حافظ على هذا السلوك الصحي.")
            awareness = "مرتفع"
        else:
            st.warning("⚠️ لديك وعي منخفض بهدر الأدوية. يُفضل مراجعة سلوكياتك لتحسينه.")
            awareness = "منخفض"

        input_dict = {}
        for i in range(len(display_questions)):
            input_dict[feature_cols[i]] = user_responses[i]
        for i in range(len(display_questions), len(feature_cols)):
            input_dict[feature_cols[i]] = 3

        data_to_save = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "student_id": student_id,
            "responses": input_dict,
            "awareness_level": awareness,
            "average_score": avg_score
        }
        save_data(data_to_save)
