# =========================================================
# MindSense AI - FINAL HYBRID AI MODEL
# =========================================================

# pip install gradio scikit-learn pandas numpy

# =========================================================
# IMPORTS
# =========================================================
import gradio as gr
import pandas as pd
import numpy as np
import os

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# =========================================================
# LOAD DATASET
# =========================================================
DATA_PATH = "Mental Health Disorder Detection Dataset.csv"

df = pd.read_csv(DATA_PATH)

df = df[['body', 'category']].dropna()

df = df[df['body'].str.len() > 20]

df['target'] = np.where(
    df['category'] == 'Depression',
    'Depression',
    'Other'
)

# =========================================================
# TRAIN MODEL
# =========================================================
X = df['body'].astype(str)
y = df['target'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words='english'
    )),

    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

model.fit(X_train, y_train)

# =========================================================
# CREATE CSV FILE
# =========================================================
CSV_FILE = "prediction_history.csv"

if not os.path.exists(CSV_FILE):

    history_df = pd.DataFrame(columns=[
        "Timestamp",
        "Name",
        "Age",
        "Gender",
        "Input Text",
        "Prediction",
        "Confidence"
    ])

    history_df.to_csv(CSV_FILE, index=False)

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_mental_health(name, age, gender, text):

    # =====================================================
    # VALIDATION
    # =====================================================
    if len(text.strip()) < 25:

        return (
            "⚠ Please enter a more detailed sentence.",
            "Try writing at least 1-2 complete emotional sentences.",
            "🤔 Insufficient Context"
        )

    # =====================================================
    # HYBRID SENTIMENT + ML PREDICTION
    # =====================================================
    text_lower = text.lower()

    positive_words = [
        "happy",
        "great",
        "good",
        "excited",
        "joy",
        "fine",
        "motivated",
        "better",
        "relaxed",
        "peaceful",
        "awesome",
        "positive",
        "energetic",
        "hopeful",
        "grateful",
        "blessed",
        "enjoying"
    ]

    negative_words = [
        "sad",
        "depressed",
        "lonely",
        "stress",
        "stressed",
        "anxiety",
        "hopeless",
        "tired",
        "crying",
        "suicidal",
        "upset",
        "worthless",
        "depression",
        "fear"
    ]

    positive_score = sum(
        word in text_lower for word in positive_words
    )

    negative_score = sum(
        word in text_lower for word in negative_words
    )

    # =====================================================
    # FINAL PREDICTION
    # =====================================================
    if positive_score > negative_score:

        prediction = "Other"

        confidence = 90.0

    else:

        prediction = model.predict([text])[0]

        prob = model.predict_proba([text])[0]

        confidence = round(np.max(prob) * 100, 2)

    # =====================================================
    # SAVE TO CSV
    # =====================================================
    new_data = pd.DataFrame([{
        "Timestamp": datetime.now(),
        "Name": name,
        "Age": age,
        "Gender": gender,
        "Input Text": text,
        "Prediction": prediction,
        "Confidence": confidence
    }])

    new_data.to_csv(
        CSV_FILE,
        mode='a',
        header=False,
        index=False
    )

    # =====================================================
    # OUTPUTS
    # =====================================================
    if prediction == "Depression":

        result = f"""
🔴 Depression Detected

Confidence Score: {confidence}%

Hello {name},

The system detected emotional stress patterns.
"""

        tips = """
🌸 Wellness Suggestions

• Talk to trusted people
• Get proper sleep
• Avoid isolation
• Practice mindfulness
• Seek professional support if needed
"""

        mood = "😔 Sad / Stressed"

    else:

        result = f"""
🟢 Emotionally Stable

Confidence Score: {confidence}%

Hello {name},

The system did not detect strong depressive patterns.
"""

        tips = """
🌟 Positive Suggestions

• Maintain healthy routines
• Continue positive activities
• Stay socially active
• Maintain work-life balance
"""

        mood = "😊 Positive / Stable"

    return result, tips, mood

# =========================================================
# CUSTOM CSS
# =========================================================
custom_css = """

body {
    background: linear-gradient(135deg, #FDF2F8, #E0F2FE);
}

.gradio-container {
    background: linear-gradient(135deg, #FDF2F8, #E0F2FE);
    font-family: 'Poppins', sans-serif;
}

textarea {
    border-radius: 18px !important;
    border: 2px solid #D8B4FE !important;
    background: white !important;
}

input {
    border-radius: 15px !important;
    border: 2px solid #D8B4FE !important;
    background: white !important;
}

select {
    border-radius: 15px !important;
    border: 2px solid #D8B4FE !important;
    background: white !important;
    height: 45px !important;
}

button {
    background: linear-gradient(to right, #EC4899, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 18px !important;
    height: 60px !important;
    font-size: 22px !important;
    font-weight: bold !important;
}

.block {
    border-radius: 24px !important;
    background: rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

footer {
    visibility: hidden;
}

"""

# =========================================================
# UI
# =========================================================
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft()
) as app:

    # =====================================================
    # HEADER
    # =====================================================
    gr.HTML("""
    <div style="
        text-align:center;
        padding:25px;
        border-radius:25px;
        background:rgba(255,255,255,0.75);
        margin-bottom:20px;
    ">

    <h1 style="
        margin:0;
        font-size:65px;
        font-weight:800;
        background: linear-gradient(to right, #EC4899, #8B5CF6);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
    ">
    🧠 MindSense AI
    </h1>

    <p style="
        margin-top:10px;
        font-size:28px;
        font-weight:600;
        color:#374151;
    ">
    AI-Based Mental Health Detection System
    </p>

    <p style="
        margin-top:8px;
        font-size:18px;
        color:#6B7280;
    ">
    Analyze emotional text using Machine Learning and NLP
    </p>

    </div>
    """)

    # =====================================================
    # MAIN SECTION
    # =====================================================
    with gr.Row(equal_height=True):

        # =================================================
        # LEFT SIDE
        # =================================================
        with gr.Column(scale=1):

            name = gr.Textbox(
                label="👤 Your Name",
                placeholder="Enter your name"
            )

            age = gr.Number(
                label="🎂 Age",
                value=20,
                precision=0
            )

            gender = gr.Radio(
                choices=["Male", "Female", "Other"],
                label="⚧ Gender",
                value="Male"
            )

            user_input = gr.Textbox(
                lines=12,
                label="📝 Enter Your Thoughts",
                placeholder="""
Example:

I have been feeling mentally exhausted and lonely lately.
I struggle to stay motivated and often feel stressed.
                """
            )

            predict_btn = gr.Button(
                "✨ Analyze Mental Health"
            )

        # =================================================
        # RIGHT SIDE
        # =================================================
        with gr.Column(scale=1):

            mood_output = gr.Textbox(
                label="🌈 Mood Indicator"
            )

            prediction_output = gr.Textbox(
                label="🧠 Prediction Result",
                lines=8
            )

            tips_output = gr.Textbox(
                label="💡 Wellness Suggestions",
                lines=10
            )

    # =====================================================
    # BUTTON CLICK
    # =====================================================
    predict_btn.click(
        fn=predict_mental_health,
        inputs=[
            name,
            age,
            gender,
            user_input
        ],
        outputs=[
            prediction_output,
            tips_output,
            mood_output
        ]
    )

    # =====================================================
    # FOOTER
    # =====================================================
    gr.HTML("""
    <div style="
        margin-top:30px;
        padding:30px;
        border-radius:25px;
        background:rgba(255,255,255,0.75);
        text-align:center;
    ">

    <h2 style="
        font-size:38px;
        font-weight:700;
        margin-bottom:20px;
        color:#1F2937;
    ">
    🌸 About MindSense AI
    </h2>

    <p style="
        font-size:20px;
        color:#4B5563;
        margin-bottom:20px;
    ">
    MindSense AI uses:
    </p>

    <div style="
        display:flex;
        justify-content:center;
        gap:35px;
        flex-wrap:wrap;
        margin-bottom:25px;
    ">

    <div style="
        background:white;
        padding:15px 25px;
        border-radius:15px;
        font-size:18px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    ">
    🧠 NLP
    </div>

    <div style="
        background:white;
        padding:15px 25px;
        border-radius:15px;
        font-size:18px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    ">
    🤖 Machine Learning
    </div>

    <div style="
        background:white;
        padding:15px 25px;
        border-radius:15px;
        font-size:18px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    ">
    📊 TF-IDF
    </div>

    <div style="
        background:white;
        padding:15px 25px;
        border-radius:15px;
        font-size:18px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    ">
    💻 Gradio UI
    </div>

    </div>

    <p style="
        font-size:18px;
        color:#6B7280;
        margin-bottom:25px;
    ">
    to analyze emotional stress patterns from text.
    </p>

    <hr style="
        border:none;
        border-top:1px solid #D1D5DB;
        margin:25px 0;
    ">

    <p style="
        font-size:18px;
        color:#374151;
        font-weight:600;
    ">
    👩‍💻 Developed By Nidhita Kochhar
    </p>

    </div>
    """)

# =========================================================
# LAUNCH APP
# =========================================================
app.launch(share=True)
