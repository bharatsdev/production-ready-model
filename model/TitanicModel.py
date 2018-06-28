# Libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


class PreProcessing:
    def __init__(self):
        pass


def build_n_train_model():
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    print(train_df.columns.values)
    # print(train_df.head())
    # print(train_df.tail())
    # print(train_df.info())
    # print(train_df.describe(include=['O']))
    Pclass_Servived = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False)
    # print(Pclass_Servived)

    # print("Before", train_df.shape, test_df.shape)
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

    # print("Af", train_df.shape, test_df.shape)

    train_df['Title'] = train_df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.', expand=False)

    train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                   'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

    test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
    test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
    test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

    # print('>>>>>>>>>> ', train_df.head())

    # We can convert the categorical to ordinal
    title_map = dict({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
    train_df.Title = train_df.Title.replace(title_map)
    train_df.Title = train_df.Title.fillna(0)

    test_df.Title = test_df.Title.replace(title_map)
    test_df.Title = test_df.Title.fillna(0)

    # print(train_df)

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    # print(train_df.columns.values)
    # print(test_df.columns.values)

    # Convert Categorical to  onehotencoding
    train_df.Sex = train_df.Sex.map({'female': 1, 'male': 0}).astype(int)
    test_df.Sex = test_df.Sex.map({'female': 1, 'male': 0}).astype(int)
    # print(train_df)

    guess_ages = np.zeros((2, 3))
    guess_ages
    combine = [train_df, test_df]
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    train_df['IsAlone'] = 0
    test_df['IsAlone'] = 0
    train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1
    test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
    # # print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
    # print(" :::::::::::::::::: ")
    # print(test_df)

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    train_df['Age*Class'] = train_df.Age * train_df.Pclass
    test_df['Age*Class'] = test_df.Age * test_df.Pclass

    freq_port = train_df.Embarked.dropna().mode()[0]
    # print(freq_port)

    train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
    test_df['Embarked'] = test_df['Embarked'].fillna(freq_port)
    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                ascending=True)

    train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0
    train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
    train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 2
    train_df.loc[train_df['Fare'] > 31, 'Fare'] = 3
    train_df['Fare'] = train_df['Fare'].astype(int)

    test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0
    test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1
    test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 2
    test_df.loc[test_df['Fare'] > 31, 'Fare'] = 3
    test_df['Fare'] = test_df['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    # X_train.shape, Y_train.shape, X_test.shape
    # print(X_train.head())
    # print(X_test.head())

    random_forest = RandomForestClassifier(n_estimators=100)
    print(X_train.shape)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    print(random_forest.score(X_train, Y_train))
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)


if __name__ == '__main__':
    model = build_n_train_model()
