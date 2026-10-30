import streamlit as st
import pandas as pd
import pickle
import re
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("spam_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# FUNCTIONS
# ============================================================

def analyze_message(message):

    word_count = len(message.split())
    character_count = len(message)

    url_count = len(
        re.findall(
            r"http\S+|www\S+|https\S+",
            message
        )
    )

    number_count = len(
        re.findall(
            r"\d",
            message
        )
    )

    special_character_count = len(
        re.findall(
            r"[^a-zA-Z0-9\s]",
            message
        )
    )

    return {
        "Words": word_count,
        "Characters": character_count,
        "URLs": url_count,
        "Numbers": number_count,
        "Special Characters": special_character_count
    }


def predict_message(message):

    message_vectorized = vectorizer.transform(
        [message]
    )

    prediction = model.predict(
        message_vectorized
    )[0]

    probabilities = model.predict_proba(
        message_vectorized
    )[0]

    not_spam_probability = probabilities[0]
    spam_probability = probabilities[1]

    if prediction == 1:
        label = "Spam"
        confidence = spam_probability
    else:
        label = "Not Spam"
        confidence = not_spam_probability

    return (
        label,
        confidence,
        not_spam_probability,
        spam_probability
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📩 Spam Classifier")

st.sidebar.write(
    "Machine Learning based SMS Spam Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Single Prediction",
        "📁 Batch Prediction",
        "📊 Model Performance",
        "🕘 Prediction History"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("📩 Spam Message Classifier")

st.caption(
    "NLP + TF-IDF + Logistic Regression"
)


# ============================================================
# SINGLE PREDICTION
# ============================================================

if page == "🏠 Single Prediction":

    st.header("🔍 Check a Message")

    st.write(
        "Enter an SMS, email, or other text message "
        "to determine whether it is Spam or Not Spam."
    )


    # Example buttons

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🧪 Try Spam Example"):

            st.session_state.example_message = (
                "Congratulations! You have won "
                "a FREE prize! Click now to claim "
                "your reward!"
            )


    with col2:

        if st.button("🧪 Try Normal Example"):

            st.session_state.example_message = (
                "Hey, are we still meeting at "
                "5 pm today?"
            )


    default_message = st.session_state.get(
        "example_message",
        ""
    )


    message = st.text_area(
        "Enter your message:",
        value=default_message,
        height=180,
        placeholder="Type or paste your message here..."
    )


    if st.button(
        "🚀 Predict Message",
        type="primary"
    ):

        if not message.strip():

            st.warning(
                "Please enter a message first."
            )

        else:

            label, confidence, not_spam, spam = (
                predict_message(message)
            )


            # Save history

            st.session_state.history.append(
                {
                    "Message": message,
                    "Prediction": label,
                    "Confidence": f"{confidence * 100:.2f}%"
                }
            )


            st.divider()


            # Result

            if label == "Spam":

                st.error(
                    f"🚨 SPAM MESSAGE"
                )

            else:

                st.success(
                    f"✅ NOT SPAM"
                )


            # Confidence

            st.subheader("Prediction Confidence")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Not Spam",
                    f"{not_spam * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Spam",
                    f"{spam * 100:.2f}%"
                )


            st.progress(
                float(confidence)
            )


            # Message analysis

            st.subheader(
                "🔎 Message Analysis"
            )

            analysis = analyze_message(
                message
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Words",
                analysis["Words"]
            )

            c2.metric(
                "Characters",
                analysis["Characters"]
            )

            c3.metric(
                "URLs",
                analysis["URLs"]
            )

            c4.metric(
                "Numbers",
                analysis["Numbers"]
            )

            c5.metric(
                "Special Characters",
                analysis["Special Characters"]
            )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📁 Batch Prediction":

    st.header("📁 Batch Message Classification")

    st.write(
        "Upload a CSV file containing messages "
        "and classify them automatically."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        data = pd.read_csv(
            uploaded_file
        )


        st.subheader(
            "Uploaded Data"
        )

        st.dataframe(
            data,
            use_container_width=True
        )


        # Find message column

        possible_columns = [
            "message",
            "text",
            "sms",
            "content"
        ]

        message_column = None

        for column in data.columns:

            if column.lower() in possible_columns:

                message_column = column
                break


        if message_column is None:

            st.error(
                "Could not find a message column. "
                "Please use a column named 'message' or 'text'."
            )

        else:

            if st.button(
                "🚀 Classify All Messages",
                type="primary"
            ):

                messages = (
                    data[message_column]
                    .fillna("")
                    .astype(str)
                )


                vectors = vectorizer.transform(
                    messages
                )


                predictions = model.predict(
                    vectors
                )


                probabilities = model.predict_proba(
                    vectors
                )


                data["Prediction"] = [
                    "Spam" if prediction == 1
                    else "Not Spam"
                    for prediction in predictions
                ]


                data["Confidence"] = [
                    max(probability) * 100
                    for probability in probabilities
                ]


                st.subheader(
                    "Classification Results"
                )


                st.dataframe(
                    data,
                    use_container_width=True
                )


                # Summary

                spam_count = (
                    data["Prediction"]
                    .eq("Spam")
                    .sum()
                )

                not_spam_count = (
                    data["Prediction"]
                    .eq("Not Spam")
                    .sum()
                )


                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Messages",
                    len(data)
                )

                c2.metric(
                    "Spam",
                    spam_count
                )

                c3.metric(
                    "Not Spam",
                    not_spam_count
                )


                # Download

                csv = data.to_csv(
                    index=False
                )


                st.download_button(
                    "⬇️ Download Results",
                    csv,
                    "spam_predictions.csv",
                    "text/csv"
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Evaluation results from the trained "
        "Spam Message Classifier."
    )


    st.info(
        "The evaluation metrics below should be "
        "updated with the actual results from train.py."
    )


    if os.path.exists(
        "confusion_matrix.png"
    ):

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            "confusion_matrix.png",
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "🕘 Prediction History":

    st.header("🕘 Prediction History")

    if len(
        st.session_state.history
    ) == 0:

        st.info(
            "No predictions have been made yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )


        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.history = []

            st.rerun()