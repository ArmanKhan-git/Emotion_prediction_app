import streamlit as st
import joblib
import pandas as pd

# =====================
# Load model, vectorizer, and mapping
# =====================
model = joblib.load("model_LR.pkl")
vectorizer = joblib.load("vectorizer_TF.pkl")
emotion_mapping = joblib.load("emotion_mapping.pkl")   # your saved dict { "happy":0, "sad":1, ... }

# Reverse mapping {0:"happy", 1:"sad", ...}
reverse_mapping = {v: k for k, v in emotion_mapping.items()}

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="centered")

st.title("🎭 Text Emotion Classifier")
st.write("Enter any text below and let the model predict the **emotion**.")

# User input
user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Transform input
        X_input = vectorizer.transform([user_input])

        # Predict numeric class and probabilities
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        # Convert numeric prediction to label
        emotion_label = reverse_mapping[prediction]

        # =====================
        # Show Results
        # =====================
        st.subheader("📌 Prediction")
        st.success(f"Predicted Emotion: **{emotion_label}**")

        # Create DataFrame for probabilities
        prob_dict = {reverse_mapping[cls]: prob for cls, prob in zip(model.classes_, probabilities)}
        prob_df = pd.DataFrame(prob_dict.items(), columns=["Emotion", "Probability"])
        prob_df = prob_df.sort_values(by="Probability", ascending=False)

        st.subheader("🔮 Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Emotion"))
