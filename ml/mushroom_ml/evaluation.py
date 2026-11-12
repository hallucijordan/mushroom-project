from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(learner, X, y, cv=5):
    cv_scores = cross_val_score(learner._model, X, y, cv=cv, scoring='accuracy')
    return {
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }