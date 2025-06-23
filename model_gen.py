import argparse
import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from datasets import load_dataset
import numpy as np
import re

MODEL_PATH = 'content_inspector.pkl'

def get_dataset(n_samples=200):
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    rows = [row for row in ds if row['human_answers'] and row['chatgpt_answers']]
    humans, ais = [], []
    for row in rows:
        if len(humans) < n_samples//2 and row['human_answers']:
            txt = row['human_answers'][0].strip()
            if len(txt) > 40:
                humans.append({'text': txt, 'label': 0})
        if len(ais) < n_samples//2 and row['chatgpt_answers']:
            txt = row['chatgpt_answers'][0].strip()
            if len(txt) > 40:
                ais.append({'text': txt, 'label': 1})
        if len(humans) >= n_samples//2 and len(ais) >= n_samples//2:
            break
    all_samples = humans + ais
    X = [item['text'] for item in all_samples]
    y = [item['label'] for item in all_samples]
    return X, y

class FeatureStats:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = []
        for text in X:
            text = text.strip()
            length = len(text)
            word_count = len(text.split())
            avg_word_len = np.mean([len(w) for w in text.split()]) if word_count else 0
            sent_count = len(re.findall(r'[.!?]', text))
            stopwords = len([w for w in text.lower().split() if w in STOPWORDS])
            uppercase = sum(1 for c in text if c.isupper())
            punctuation = sum(1 for c in text if c in ".,;:!?")
            digit_count = sum(1 for c in text if c.isdigit())
            feats.append([
                length,
                word_count,
                avg_word_len,
                sent_count,
                stopwords,
                uppercase,
                punctuation,
                digit_count
            ])
        return np.array(feats)

STOPWORDS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don don should now
""".split())

def create_pipeline():
    tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.8, sublinear_tf=True)
    union = FeatureUnion([
        ('tfidf', tfidf),
        ('stats', FeatureStats())
    ])
    pipe = Pipeline([
        ('feats', union),
        ('clf', LogisticRegression(solver='liblinear', C=5, max_iter=1000))
    ])
    return pipe

def train():
    X, y = get_dataset(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = create_pipeline()
    pipe.fit(X_train, y_train)
    joblib.dump({'pipeline': pipe, 'X_test': X_test, 'y_test': y_test}, MODEL_PATH)
    print("[+] Model trained and saved to '%s'" % MODEL_PATH)

def predict(text):
    try:
        data = joblib.load(MODEL_PATH)
        pipe = data['pipeline']
    except:
        print("[-] Trained model not found. Please run with --train first.")
        sys.exit(1)
    proba = pipe.predict_proba([text])[0]
    label = pipe.predict([text])[0]
    conf = proba[label]
    color = '\033[92m' if label == 0 else '\033[91m'
    reset = '\033[0m'
    label_str = f"{color}{'Human' if label == 0 else 'AI'}{reset}"
    print(f"Prediction: {label_str} (confidence: {conf*100:.1f}%)")

def evaluate():
    try:
        data = joblib.load(MODEL_PATH)
        pipe = data['pipeline']
        X_test = data['X_test']
        y_test = data['y_test']
    except:
        print("[-] Trained model not found. Please run with --train first.")
        sys.exit(1)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

def main():
    parser = argparse.ArgumentParser(description="AI-powered Content Inspector")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', type=str, metavar='"your text"')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.train:
        train()
    elif args.predict is not None:
        predict(args.predict)
    elif args.evaluate:
        evaluate()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
