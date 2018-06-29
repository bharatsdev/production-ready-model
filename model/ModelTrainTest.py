# Libraries
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
# from processing.PreProcessing import PreProcessing
from sklearn.base import BaseEstimator, TransformerMixin
import pickle as pk


class PreProcessing(BaseEstimator, TransformerMixin):
    """" This Class is our customized Pre-Process """

    def __init__(self):
        pass

    def transform(self, df, y=None, **fit_params):
        print("************************************************* Before Transform ***************************")
        print(df.columns[df.isna().any()].tolist())
        print(df.dtypes)
        """ This is our customized transform method will be use for Train, Test and Validation data transform"""
        # print("Check the Corellation")
        # Pclass_Servived = df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(
        #     by='Survived', ascending=False)
        df = df.drop(['Ticket', 'Cabin'], axis=1)
        # print("Af", df.shape)
        df['Title'] = self.Title.astype(int)
        df['Title'] = df['Title'].fillna(0).astype(int)
        # print("Af", df.shape)
        # print(df.head())
        # print("<<<<<<<<<<<<<>>> ", df.columns.values)
        df = df.drop(self.drop_Name_features, axis=1)
        df.Sex = df.Sex.map(self.sex_map).astype(int)
        df['FamilySize'] = self.FamilySize
        df['IsAlone'] = self.df_IsAlone
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        df.IsAlone = df.IsAlone.fillna(0)
        df = df.drop(self.drop_Parch_SibSp_FamilySize, axis=1)
        df['Embarked'] = df.Embarked.fillna(self.freq_port)
        df['Embarked'] = df['Embarked'].map(self.Embarked_map).astype(int)
        df['Fare'].fillna(self.Fare_median, inplace=True)
        df['FareBand'] = self.Fare_qcut
        df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                              ascending=True)
        df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
        df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
        df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
        df.loc[df['Fare'] > 31, 'Fare'] = 3
        df['Fare'] = df['Fare'].astype(int)
        df = df.drop(['FareBand'], axis=1)
        df.Age = df.Age.fillna(0).astype(int)

        target_feature = ['Survived']
        df = df.drop(target_feature, axis=1)
        # print("Af >> ", df.shape)
        # print("Drop ", df.columns.values)
        # print("Af >> ", df.dtypes)
        # print(df.columns[df.isna().any()].tolist())
        # print("***************** After Transform **********************")
        # print(df.columns.values)
        print("************************************************* After Transform ***************************")
        print(df.columns[df.isna().any()].tolist())
        print(df.dtypes)
        return df.as_matrix()
        # print(df)

    def fit(self, df, y=None, **fit_params):
        # print("***************** Before Fit **********************")
        self.Title = df.Name.str.extract('([A-Za-z]+)\.', expand=False)
        # print(self.Title)
        title_rare = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        self.Title = self.Title.replace(title_rare, 'Rare')
        self.Title = self.Title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        # print(self.Title)
        # We can convert the categorical to ordinal
        title_map = dict({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
        self.Title = self.Title.replace(title_map)
        self.Title = self.Title.fillna(0)
        self.drop_Name_features = ['Name', 'PassengerId']
        # Convert Categorical to  onehotencoding
        self.sex_map = dict({'female': 1, 'male': 0})
        self.FamilySize = df.SibSp + df.Parch + 1
        self.df_IsAlone = pd.DataFrame(columns=['IsAlone'])
        self.df_IsAlone = self.df_IsAlone.fillna(0)
        self.drop_Parch_SibSp_FamilySize = ['Parch', 'SibSp', 'FamilySize']
        self.freq_port = df.Embarked.dropna().mode()[0]
        self.Embarked_map = {'S': 0, 'C': 1, 'Q': 2}
        self.Fare_median = df['Fare'].dropna().median()
        self.Fare_qcut = pd.qcut(df['Fare'], 4)
        self.age_mising = df.Age.median()
        # print("***************** After Fit **********************")

        # print(self.Title)
        return self


def build_n_train_model():
    train_df = pd.read_csv('../data/train.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(train_df, train_df['Survived'], test_size=0.25,
                                                        random_state=41)
    # print(X_train.columns[X_train.isna().any()].tolist())
    pipe = make_pipeline(PreProcessing(), RandomForestClassifier(n_estimators=100))
    aram_grids = {"randomforestclassifier__n_estimators": [10, 20, 30],
                  "randomforestclassifier__max_depth": [None, 6, 8, 10],
                  "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20]}
    grid = GridSearchCV(pipe, param_grid=aram_grids, cv=3)
    grid.fit(X_train, Y_train)

    print(grid.scorer_)
    print(grid.best_estimator_)

    y_pred_train = grid.predict(X_train)
    print("Training Score ", accuracy_score(Y_train, y_pred_train))
    y_pred_test = grid.predict(X_test)
    print("Testing Score ", accuracy_score(Y_test, y_pred_test))
    return grid


if __name__ == '__main__':
    model = build_n_train_model()
    fileName = '../data/aitechwizard.pkl'
    pickle = pk.dump(model, open(fileName, "wb"))
    test_df = pd.read_csv('../data/test.csv')
    test_df['Survived'] = 0
    # load the Pickle Model from
    print("Testing ::::::::::::::::::::: ")
    clf = pk.load(open(fileName, "rb"))
    clf.predict(test_df)
