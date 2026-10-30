# SMS Spam Classifier

A Machine Learning-based web application that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP).

## Project Overview

The SMS Spam Classifier helps identify unwanted or promotional messages. Users can enter an SMS message into the application and receive a prediction about whether it is spam or legitimate.

![SMS Spam Classifier Demo](spam-classifier-demo.png)

## Features

- SMS message classification
- Machine Learning-based prediction
- Text preprocessing and vectorization
- Interactive Streamlit web interface
- Pre-trained model for faster predictions
- Confusion matrix for model evaluation

## Tech Stack

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Pandas
- Scikit-learn
- Streamlit

## Project Structure

```text
SMS-Spam-Classifier/
│
├── app.py
├── train.py
├── check_data.py
├── requirements.txt
├── spam_model.pkl
├── vectorizer.pkl
├── confusion_matrix.png
├── data/
└── templates/
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/Raadhiikaa1/SMS-Spam-Classifier.git
```

2. Navigate to the project directory:

```bash
cd SMS-Spam-Classifier
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Open the local URL displayed in your terminal.

## How It Works

1. The user enters an SMS message.
2. The message is processed and converted into numerical features using the saved vectorizer.
3. The trained Machine Learning model analyses the features.
4. The application displays the prediction: Spam or Not Spam.

## Model Evaluation

The project includes a confusion matrix to help visualize the classification performance.

## Future Improvements

- Improve model accuracy.
- Add a prediction history feature.
- Enhance the user interface.
- Support bulk SMS classification.

## Author

**Radhika Sharma**

GitHub: https://github.com/Raadhiikaa1

---

*This project was developed as a hands-on Machine Learning project to explore text classification and NLP.*
