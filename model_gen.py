import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datasets import load_dataset

MODEL_PATH = 'content_inspector.pkl'

def get_dataset(n_samples=2500):
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

def create_pipeline():
    tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.8, sublinear_tf=True)
    pipe = Pipeline([
        ('tfidf', tfidf),
        ('clf', LogisticRegression(solver='liblinear', C=10, max_iter=2000))
    ])
    return pipe

def train():
    X, y = get_dataset(n_samples=2500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = create_pipeline()
    pipe.fit(X_train, y_train)
    joblib.dump({'pipeline': pipe, 'X_test': X_test, 'y_test': y_test}, MODEL_PATH)
    print("[+] Model trained and saved to '%s'" % MODEL_PATH)

def evaluate():
    data = joblib.load(MODEL_PATH)
    pipe = data['pipeline']
    X_test = data['X_test']
    y_test = data['y_test']
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.train:
        train()
    if args.evaluate:
        evaluate()
