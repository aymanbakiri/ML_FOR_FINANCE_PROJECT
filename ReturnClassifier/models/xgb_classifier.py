from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def build_xgb_pipeline():
    pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),  
    ('clf',      XGBClassifier(
         objective='binary:logistic',
         eval_metric='auc',
         use_label_encoder=False,
         n_estimators=200,
         max_depth=5,
         learning_rate=0.05,
         random_state=42
    ))
])
    return pipe, {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.05]
}

