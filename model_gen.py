import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datasets import load_dataset

MODEL_PATH = 'content_inspector_combo.pkl'

def get_dataset(n_samples=2000):
    ds_hc3 = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    ds_ai_vs_human = load_dataset("zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-360M-Instruct", split="train")
    samples = []
    for row in ds_hc3:
        if row['human_answers'] and len(samples) < n_samples:
            txt = row['human_answers'][0].strip()
            if len(txt) > 40:
                samples.append({'text': txt, 'label': 0})
        if row['chatgpt_answers'] and len(samples) < 2*n_samples:
            txt = row['chatgpt_answers'][0].strip()
            if len(txt) > 40:
                samples.append({'text': txt, 'label': 1})
        if len(samples) >= 2*n_samples:
            break
    for row in ds_ai_vs_human:
        txt_ai = row['ai'].strip()
        txt_human = row['human'].strip()
        if len(txt_ai) > 40:
            samples.append({'text': txt_ai, 'label': 1})
        if len(txt_human) > 40:
            samples.append({'text': txt_human, 'label': 0})
        if len(samples) >= 4*n_samples:
            break
    X = [item['text'] for item in samples[:2*n_samples]]
    y = [item['label'] for item in samples[:2*n_samples]]
    return X, y

def create_pipeline():
    tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.8, sublinear_tf=True)
    pipe = Pipeline([
        ('tfidf', tfidf),
        ('clf', LogisticRegression(solver='liblinear', C=10, max_iter=2000))
    ])
    return pipe

def train():
    X, y = get_dataset(n_samples=2000)
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
