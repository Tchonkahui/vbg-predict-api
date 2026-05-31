# ============================================================
# ENTRAÎNEMENT LOCAL DU MODÈLE — VBG
# ============================================================

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Chargement
print("📂 Chargement du dataset...")
df = pd.read_csv('Domestic violence.csv')

# Nettoyage
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df = df.drop(columns=['sl._no'], errors='ignore')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()
df['marital_status'] = df['marital_status'].replace('unmarred','unmarried')

# Feature Engineering
df['has_income']  = (df['income'] > 0).astype(int)
df['income_log']  = np.log1p(df['income'])
df['age_group']   = pd.cut(df['age'],
    bins=[15,25,35,45,60],
    labels=['15-24','25-34','35-44','45-60'],
    include_lowest=True).astype(str)
edu_risk = {'no education':2,'primary':1,'secondary':0,'tertiary':0}
df['edu_risk']    = df['education'].map(edu_risk).fillna(0)
df['emp_risk']    = (df['employment']=='unemployed').astype(int)
df['mar_risk']    = (df['marital_status']=='married').astype(int)
df['inc_risk']    = (df['income']==0).astype(int)
df['age_risk']    = df['age'].apply(lambda x: 1 if 25<=x<=35 else 0)
df['vulnerability_score'] = (df['edu_risk']+df['emp_risk']+
                              df['mar_risk']+df['inc_risk']+df['age_risk'])
df['edu_x_emp']   = df['education']+'_'+df['employment']

# X, y
y = (df['violence']=='yes').astype(int)
features_num = ['age','income_log','vulnerability_score']
features_bin = ['has_income','emp_risk','mar_risk','inc_risk','age_risk','edu_risk']
features_ord = ['education']
features_nom = ['employment','marital_status','age_group']
features_int = ['edu_x_emp']
all_features = features_num+features_bin+features_ord+features_nom+features_int
X = df[all_features].copy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
num_pipe = Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('scaler',StandardScaler())])
bin_pipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent'))])
ord_pipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                     ('enc',OrdinalEncoder(
                         categories=[['no education','primary','secondary','tertiary']],
                         handle_unknown='use_encoded_value',unknown_value=-1))])
nom_pipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                     ('enc',OneHotEncoder(drop='first',handle_unknown='ignore',
                                          sparse_output=False))])
int_pipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                     ('enc',OneHotEncoder(drop='first',handle_unknown='ignore',
                                          sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num',num_pipe,features_num),
    ('bin',bin_pipe,features_bin),
    ('ord',ord_pipe,features_ord),
    ('nom',nom_pipe,features_nom),
    ('int',int_pipe,features_int),
], remainder='drop')

model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.6, k_neighbors=5, random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_leaf=5, class_weight='balanced',
        random_state=42, n_jobs=-1))
])

# Entraînement
print("⏳ Entraînement du modèle...")
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
recall = recall_score(y_test, y_pred)
print(f"\n✅ Modèle entraîné")
print(f"Recall : {recall:.3f}")
print(classification_report(y_test, y_pred,
      target_names=['Non-Violence','Violence']))

# Sauvegarde
joblib.dump(model, 'model_vbg.pkl')
with open('model_metadata.json', 'w') as f:
    json.dump({
        "nom_modele" : "Random Forest",
        "version"    : "1.0.0",
        "date"       : datetime.now().isoformat(),
        "features"   : all_features,
    }, f, indent=2, ensure_ascii=False)

print("💾 Modèle sauvegardé : model_vbg.pkl")