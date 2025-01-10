################################################
# Exploratory Data Analysis
################################################
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.simplefilter(action='ignore', category=Warning)


train = pd.read_csv("datasets/train.csv")
train.head()
test = pd.read_csv("datasets/test.csv")
df = pd.concat([train, test], ignore_index=True)
df.head()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10,  car_th=20):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int64", "float64"]]
    num_but_cat = [col for col in num_but_cat if col not in "YrSold"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() >= car_th and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

# Observations: 2919
# Variables: 81
# cat_cols: 51    ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour','Utilities', 'LotConfig',
#                  'LandSlope', 'Condition1','Condition2', 'BldgType', 'HouseStyle','RoofStyle',
#                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
#                  'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#                  'Heating','HeatingQC', 'CentralAir','Electrical', 'KitchenQual', 'Functional',
#                  'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','PavedDrive',
#                  'PoolQC','Fence','MiscFeature', 'SaleType','SaleCondition', 'OverallCond', 'BsmtFullBath',
#                  'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','Fireplaces','GarageCars']
# num_cols: 29    ['Id','MSSubClass','LotFrontage','LotArea','OverallQual','YearBuilt','YearRemodAdd',
#                  'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
#                  '2ndFlrSF','LowQualFinSF','GrLivArea','TotRmsAbvGrd','GarageYrBlt','GarageArea',
#                  'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
#                  'MiscVal','MoSold','YrSold','SalePrice']
# cat_but_car: 1   ['Neighborhood']
# num_but_cat: 9   ['OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#                   'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars']

df.loc[df["GarageYrBlt"] == 2207, "GarageYrBlt"] = 2007



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)



def target_summary_with_cat(dataframe, target, cat_cols):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_cols)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)


def target_summary_with_num(dataframe, target, num_cols):
    print(dataframe.groupby(target).agg({num_cols: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "SalePrice", col)

corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))



def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "SalePrice", na_cols)


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean()*100, 2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
plt.show(block=True)

################################################
# Feature Engineering
################################################
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
             'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']
freq_cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
             'SaleType', 'Utilities']

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)
for col in none_cols:
    df[col].replace(np.nan, "None", inplace=True)
for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))
df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

missing_values_table(df)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", cat_cols)



todays_date = date.today()
todays_date.year

df["NEW_area"] = df['1stFlrSF']+df["2ndFlrSF"]
df["PropAge"] = todays_date.year - df["YearBuilt"]
df["PropAgeType"] = pd.qcut(df["PropAge"], 4, labels=["New", "Middle", "Old", "TooOld"])
df["SoldYr"] = df["YrSold"] - df["YearBuilt"]
df["HouseDemand"] = pd.qcut(df["SoldYr"], 4, labels=["HighDemand", "NormalDemand", "LessDemand", "LowDemand"])
df["GarageAge"] = df["GarageYrBlt"] - df["YearBuilt"]
df["TotalBath"] = (df["BsmtHalfBath"] + df["HalfBath"]) * 0.5 + df["BsmtFullBath"] + df["FullBath"]
df["TotalFullBath"] = df["FullBath"] + df["BsmtFullBath"]
df["TotalHalfBath"] = df["HalfBath"] + df["BsmtHalfBath"] * 0.5
df["GarageRatio"] = df["GarageArea"] / df["LotArea"] * 100
df["LivLotRatio"] = df["GrLivArea"] / df["LotArea"]
df["LotShape"].replace({"Reg": 4, "IR1": 3, "IR2": 2, "Rare": 1}, inplace=True)
df["LandSlope"].replace({"Gtl": 3, "Mod": 2, "Rare": 1}, inplace=True)
df["ExterQual"].replace({"TA": 4, "Gd": 3, "Ex": 2, "Fa": 1}, inplace=True)
df["ExterCond"].replace({"TA": 4, "Gd": 3, "Fa": 2, "Rare": 1}, inplace=True)
df["BsmtQual"].replace({"TA": 5, "Gd": 4, "Ex": 3, "Fa": 2, "None": 1}, inplace=True)
df["BsmtCond"].replace({"TA": 5, "Gd": 4, "Fa": 3, "None": 2, "Rare": 1}, inplace=True)
df["BsmtExposure"].replace({"Gd": 5, "Av": 4, "Mn": 3, "None": 2, "No": 1}, inplace=True)
df["BsmtFinType1"].replace({"GLQ": 7, "ALQ": 6, "BLQ": 5, "Rec": 4, "LwQ": 3, "Unf": 2, "None": 1}, inplace=True)
df["HeatingQC"].replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}, inplace=True)
df["KitchenQual"].replace({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1}, inplace=True)
df["HeatingQC"].replace({"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}, inplace=True)
df["FireplaceQu"].replace({"Ex": 6, "Gd": 5, "TA": 4, "Fa": 3, "Po": 2, "None": 1}, inplace=True)
df["GarageFinish"].replace({"Fin": 4, "RFn": 3, "Unf": 2, "None": 1}, inplace=True)



binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 30 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


################################################
# Model 
################################################
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# KNN
knn_model = KNeighborsRegressor().fit(X, y)
knn_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(knn_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.22290477700269803
knn_params = {"n_neighbors": range(2, 50)}
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
knn_gs_best.best_params_
# {'n_neighbors': 7}
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(knn_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.2216060184821133

# CART
cart_model = DecisionTreeRegressor(random_state=1).fit(X, y)
cart_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(cart_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.22162808001169862
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)
cart_best_grid.best_params_
# {'max_depth': 16, 'min_samples_split': 26}
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.19922352950838942

# Random Forests
rf_model = RandomForestRegressor(random_state=17)
rf_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(rf_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.14811258274538042
rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [200, 300, 500]}
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
# {'max_depth': None, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 300}
rf_final = rf_model.set_params(**rf_best_grid.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.14692292876487129

# GBM
gbm_model = GradientBoostingRegressor(random_state=17)
gbm_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(gbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.13568290580687797
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_
# {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.12689667645179953

# XGBoost
xgboost_model = XGBRegressor(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(xgboost_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.14486079212337866
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_
# {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(xgboost_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.1352015604327077

# LightGBM
lgbm_model = LGBMRegressor(random_state=17)
lgbm_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.13595909905379694
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
# {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 1000}
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.13145978592192725


def base_models(X, y, scoring="neg_mean_squared_error"):
    print("Base Models....")
    regressors = [('KNN', KNeighborsRegressor()),
                  ("CART", DecisionTreeRegressor()),
                  ("RF", RandomForestRegressor()),
                  ('GBM', GradientBoostingRegressor()),
                  ('XGBoost', XGBRegressor(objective="reg:squarederror")),
                  ('LightGBM', LGBMRegressor())]
    for name, regressor in regressors:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"rmse: {round(rmse, 4)} ({name}) ")


base_models(X, y, scoring="neg_mean_squared_error")
# obefore assigning 
# Base Models....
# rmse: 0.2229 (KNN)
# rmse: 0.2226 (CART)
# rmse: 0.1464 (RF)
# rmse: 0.1365 (GBM)
# rmse: 0.1405 (XGBoost)
# rmse: 0.1358 (LightGBM)

# after assigning
# Base Models....
# rmse: 0.2229 (KNN)
# rmse: 0.2141 (CART)
# rmse: 0.147 (RF)
# rmse: 0.1351 (GBM)
# rmse: 0.1449 (XGBoost)
# rmse: 0.136 (LightGBM)


def final_models(X, y, scoring="neg_mean_squared_error"):
    print("Final Models....")
    finalmodels = [('KNN', knn_final),
                   ("CART", cart_final),
                   ("RF", rf_final),
                   ('GBM', gbm_final),
                   ('XGBoost', xgboost_final),
                   ('LightGBM', lgbm_final)]
    for name, model in finalmodels:
        rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"rmse: {round(rmse, 4)} ({name}) ")


final_models(X, y, scoring="neg_mean_squared_error")
# bedore assigning
# Final Models....
# rmse: 0.2216 (KNN)
# rmse: 0.1999 (CART)
# rmse: 0.1464 (RF)
# rmse: 0.13 (GBM)
# rmse: 0.1369 (XGBoost)
# rmse: 0.131 (LightGBM)

# after assigning
# Final Models....
# rmse: 0.2216 (KNN)
# rmse: 0.1992 (CART)
# rmse: 0.1469 (RF)
# rmse: 0.1269 (GBM)
# rmse: 0.1352 (XGBoost)
# rmse: 0.1315 (LightGBM)

from catboost import CatBoostRegressor
catboost_model = CatBoostRegressor(random_state=17, verbose=False)
catboost_model.get_params()
rmse = np.mean(np.sqrt(-cross_val_score(catboost_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.12403518750370349
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.12373608070807969



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X, 10)
plot_importance(rf_final, X, 10)
plot_importance(gbm_final, X, 10)
plot_importance(xgboost_final, X, 10)
plot_importance(lgbm_final, X, 10)

