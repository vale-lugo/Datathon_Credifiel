import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor

# Leer clusters
cluster_files = glob.glob('cluster_*.csv')
clusters_dict = {file.split('.')[0]: pd.read_csv(file) for file in cluster_files}
df_total = pd.concat(clusters_dict.values(), ignore_index=True)

# Columnas reales (según tus imágenes)
features = [
    'idListaCobro', 'idCredito', 'consecutivoCobro', 'idBanco',
    'montoExigible', 'montoCobrar', 'idRespuestaBanco', 'idEmisora',
    'TipoEnvio', 'hora_cos', 'diaEnvioCobro_cos', 'diaCreacion_cos'
]

target = ['montoCobrado', 'costo_transaccion']

X = df_total[features]
y = df_total[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'device': 'gpu',
        'objective': 'regression'
    }

    model = LGBMRegressor(**params, n_estimators=1000)
    model.fit(
        X_train, y_train['montoCobrado'],
        eval_set=[(X_test, y_test['montoCobrado'])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    preds = model.predict(X_test)
    rmse = ((y_test['montoCobrado'] - preds)**2).mean()**0.5
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25)

print("Mejores parámetros encontrados:", study.best_params)

# Modelo final
best_params = study.best_params
best_params.update({'objective': 'regression', 'device': 'gpu'})

final_model = LGBMRegressor(**best_params, n_estimators=1500)
final_model.fit(X_train, y_train['montoCobrado'])

# Predicciones finales y recomendaciones
df_total['pred_montoCobrado'] = final_model.predict(X)

recomendaciones = df_total.groupby('idCredito').apply(
    lambda grp: grp.loc[grp['pred_montoCobrado'].idxmax()]
).reset_index(drop=True)

resultados_finales = recomendaciones[[
    'idCredito', 'TipoEnvio', 'hora_cos', 
    'diaEnvioCobro_cos', 'diaCreacion_cos', 
    'montoCobrar', 'pred_montoCobrado', 'costo_transaccion'
]]

resultados_finales.to_csv('recomendaciones_personalizadas.csv', index=False)
