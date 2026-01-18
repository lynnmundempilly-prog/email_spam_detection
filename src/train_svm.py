from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_loader import load_data
from src.vectorizer import get_tfidf_features
from src.evaluate import evaluate_model

df = load_data("data/spam.csv")
X, _ = get_tfidf_features(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred, "Linear SVM")
