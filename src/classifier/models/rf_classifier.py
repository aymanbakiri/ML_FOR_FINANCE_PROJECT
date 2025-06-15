from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def build_rf_pipeline():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',   StandardScaler()),
        ('clf',      RandomForestClassifier(
                        random_state=42, 
                        class_weight='balanced'))
    ])
    return pipe, {
        'clf__n_estimators': [100,200],
        'clf__max_depth':    [5,7,9],
        'clf__max_features':['sqrt','log2']
    }
