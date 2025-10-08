# hw02_solution.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_and_select():
    df = pd.read_csv(URL)
    cols = ['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']
    df = df[cols]
    return df

def q1_q2(df):
    # Q1: which column has missing values
    missing = df.isnull().sum()
    print("Missing counts per column:\n", missing)
    col_with_missing = missing[missing > 0].index.tolist()
    print("Column(s) with missing values:", col_with_missing)

    # Q2: median of horsepower
    hp_median = df['horsepower'].median()
    print("Median (50%) of horsepower:", hp_median)
    return col_with_missing, hp_median

def split_data(df, seed=42):
    df_sh = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_sh)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    n_test = n - n_train - n_val
    df_train = df_sh.iloc[:n_train].reset_index(drop=True)
    df_val = df_sh.iloc[n_train:n_train+n_val].reset_index(drop=True)
    df_test = df_sh.iloc[n_train+n_val:].reset_index(drop=True)
    return df_train, df_val, df_test

def prepare_Xy(df, fill_strategy, fill_value=None):
    df2 = df.copy()
    if fill_strategy == 'zero':
        df2['horsepower'] = df2['horsepower'].fillna(0)
    elif fill_strategy == 'mean':
        df2['horsepower'] = df2['horsepower'].fillna(fill_value)
    else:
        raise ValueError("Unknown fill_strategy")
    X = df2[['engine_displacement','horsepower','vehicle_weight','model_year']].values
    y = df2['fuel_efficiency_mpg'].values
    return X, y

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge(X_train, y_train, r):
    # r is regularization strength in ridge (alpha)
    model = Ridge(alpha=r, random_state=1)
    model.fit(X_train, y_train)
    return model

def question3(df_train, df_val):
    # compute mean on training only
    mean_hp = df_train['horsepower'].mean()
    # option 1: fill with 0
    Xtr0, ytr = prepare_Xy(df_train, 'zero')
    Xval0, yval = prepare_Xy(df_val, 'zero')
    # no scaling needed for linear regression (but ok). We'll keep raw.
    model0 = train_linear(Xtr0, ytr)
    pred0 = model0.predict(Xval0)
    rmse0 = round(rmse(yval, pred0), 2)

    # option 2: fill with mean
    Xtr1, _ = prepare_Xy(df_train, 'mean', fill_value=mean_hp)
    Xval1, _ = prepare_Xy(df_val, 'mean', fill_value=mean_hp)
    model1 = train_linear(Xtr1, ytr)
    pred1 = model1.predict(Xval1)
    rmse1 = round(rmse(yval, pred1), 2)

    print("Q3: RMSE fill=0:", rmse0, " RMSE fill=mean:", rmse1)
    better = None
    if rmse0 < rmse1: better = "With 0"
    elif rmse1 < rmse0: better = "With mean"
    else: better = "Both are equally good"
    print("Q3 answer:", better)
    return rmse0, rmse1, better

def question4(df_train, df_val):
    # fill NA with 0 per question
    Xtr, ytr = prepare_Xy(df_train, 'zero')
    Xval, yval = prepare_Xy(df_val, 'zero')
    # standardize features (often helps ridge but not required)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xval_s = scaler.transform(Xval)

    rs = [0, 0.01, 0.1, 1, 5, 10, 100]
    results = {}
    for r in rs:
        model = train_ridge(Xtr_s, ytr, r)
        pred = model.predict(Xval_s)
        results[r] = round(rmse(yval, pred), 4)
    print("Q4 RMSEs (r -> RMSE):", results)
    # choose best (smallest RMSE); if multiple same, smallest r
    best_rmse = min(results.values())
    best_rs = [r for r, v in results.items() if v == best_rmse]
    best_r = min(best_rs)
    print("Q4 best r:", best_r, "with RMSE:", best_rmse)
    return results, best_r

def question5(df, seeds=list(range(10))):
    scores = []
    for s in seeds:
        df_tr, df_val, df_test = split_data(df, seed=s)
        Xtr, ytr = prepare_Xy(df_tr, 'zero')
        Xval, yval = prepare_Xy(df_val, 'zero')
        model = train_linear(Xtr, ytr)
        pred = model.predict(Xval)
        scores.append(rmse(yval, pred))
    std = round(np.std(scores), 4)
    print("Q5 RMSE scores:", [round(s,4) for s in scores])
    print("Q5 std:", std)
    return scores, std

def question6(df, seed=9):
    df_tr, df_val, df_test = split_data(df, seed=seed)
    # combine train + val
    df_train_val = pd.concat([df_tr, df_val], ignore_index=True)
    Xtrv, ytrv = prepare_Xy(df_train_val, 'zero')
    Xtest, ytest = prepare_Xy(df_test, 'zero')
    scaler = StandardScaler()
    Xtrv_s = scaler.fit_transform(Xtrv)
    Xtest_s = scaler.transform(Xtest)
    model = train_ridge(Xtrv_s, ytrv, r=0.001)
    pred = model.predict(Xtest_s)
    score = round(rmse(ytest, pred), 3)
    print("Q6 RMSE on test:", score)
    return score

def main():
    df = load_and_select()
    print("Loaded dataframe shape:", df.shape)
    # EDA: check tail
    print("fuel_efficiency_mpg describe:\n", df['fuel_efficiency_mpg'].describe())
    # Q1 Q2
    col_with_missing, hp_median = q1_q2(df)

    # split seed 42
    df_train, df_val, df_test = split_data(df, seed=42)

    # Q3
    rmse0, rmse1, better = question3(df_train, df_val)

    # Q4
    q4_results, best_r = question4(df_train, df_val)

    # Q5
    seeds = list(range(10))
    scores, std = question5(df, seeds=seeds)

    # Q6
    q6_score = question6(df, seed=9)

    # Print final multiple-choice style summary:
    print("\n--- FINAL ANSWERS (choose closest option) ---")
    # Q1
    print("Q1 column with missing values:", col_with_missing)
    print("Q2 horsepower median:", hp_median)
    print("Q3 better option:", better, "(RMSE with 0: {}, with mean: {})".format(rmse0, rmse1))
    print("Q4 best r:", best_r)
    print("Q5 std:", std)
    print("Q6 test RMSE:", q6_score)

if __name__ == "__main__":
    main()
