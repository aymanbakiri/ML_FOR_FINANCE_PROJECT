from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_rfg_pipeline():
  pipe = Pipeline([
     ('imputer', SimpleImputer(strategy='median')),
     ('scaler', StandardScaler()),
   ('reg', RandomForestRegressor(random_state=42))
 ])

  return pipe, {
     'reg__n_estimators': [100, 200],
     'reg__max_depth': [5, 7, 9],
     'reg__max_features': ['sqrt', 'log2']
 }
