import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modules.language_detector import preprocess, config

def main():
    print("--> Loading model and vectorizer...")
    model = joblib.load(config.MOD1_CLASSIFIER_PATH)
    vectorizer = joblib.load(config.MOD1_VECTORIZER_PATH)

    text = "i am sad i don't want to kill myself"
    cleaned = preprocess(text)
    print(f"Original: {text}")
    print(f"Cleaned:  {cleaned}")

    # Get the TF-IDF representation
    X = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()

    # Find features that are non-zero in our query
    nonzero_indices = X.nonzero()[1]
    query_features = [(feature_names[i], X[0, i]) for i in nonzero_indices]
    
    print("\nQuery features and TF-IDF values:")
    for name, val in query_features:
        print(f"  {name}: {val:.4f}")

    # Inspect the coefficients for the 'sw' and 'en' classes
    classes = list(model.classes_)
    sw_idx = classes.index('sw')
    en_idx = classes.index('en')

    print("\nLogistic Regression Coefficients for active features:")
    for name, val in query_features:
        feat_idx = list(feature_names).index(name)
        coef_sw = model.coef_[sw_idx][feat_idx]
        coef_en = model.coef_[en_idx][feat_idx]
        print(f"  Feature '{name}':")
        print(f"    Swahili coef: {coef_sw:.4f}")
        print(f"    English coef: {coef_en:.4f}")
        print(f"    Diff (en - sw): {coef_en - coef_sw:.4f}")

if __name__ == "__main__":
    main()
