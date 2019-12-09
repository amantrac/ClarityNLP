import pandas as pd
import os

from typing import List
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers, Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from numpy import array
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform


def retain_unseen_example_in_test(df_train: DataFrame, df_test: DataFrame) -> DataFrame:
    return df_train.merge(df_test, how='left', indicator=True).loc[lambda x: x['_merge'] != 'both']


def get_text_array(df: DataFrame, feature_list: List[str]) -> Series:
    return_df = df[feature_list[0]]
    if len(feature_list) > 1:
        for f in feature_list:
            return_df += ' ' + df[f]
    return return_df.values


def get_class_array(data_frame: DataFrame,
                    tag_category='category'
                    ) -> Series:
    y = data_frame[tag_category].values
    return y


def get_one_hot_class(data_frame: DataFrame,
                      tag_category='category',
                      encoder=LabelEncoder()) -> Series:
    y = data_frame[tag_category].values
    y_encoded = encoder.fit_transform(y)
    y_categorical_labels = to_categorical(y_encoded)

    return y_categorical_labels


def create_nn_model(dropout: float,
                    input_dim: int,
                    output_dim: int,
                    hidden_units: str,
                    ) -> Model:
    hidden_units = hidden_units.split('_')
    model = Sequential()
    for units in hidden_units:
        model.add(layers.Dense(int(units), input_dim=input_dim, activation='relu'))
        model.add(layers.Dropout(dropout))
    # final layer
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def create_logistic_model(c=1.0):
    return LogisticRegression(C=c, solver='liblinear')


def evaluate_logistic(data_train: array,
                      data_test: array,
                      columns: List[str]):
    x_train_array = get_text_array(data_train, columns)
    x_test_array = get_text_array(data_test, columns)

    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word',
                                 ngram_range=(1, 2), min_df=3, lowercase=True)
    vectorizer.fit(x_train_array)
    Xtrain = vectorizer.transform(x_train_array)
    Xtest = vectorizer.transform(x_test_array)
    Ytrain = data_train['category'].values
    Ytest = data_test['category'].values
    # LOGISTIC BASELINE DEFAULT PARAMETER
    classifier = LogisticRegression(solver='liblinear',
                                    random_state=123456789)
    classifier.fit(Xtrain, Ytrain)
    score = classifier.score(Xtest, Ytest)
    print(f"Accuracy on {columns} is {score}")


def tune_logitstic(data_train: array,
                   data_test: array,
                   columns: List[str]):
    x_train_array = get_text_array(data_train, columns)
    x_test_array = get_text_array(data_test, columns)

    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word',
                                 ngram_range=(1, 2), min_df=3, lowercase=True)
    vectorizer.fit(x_train_array)
    Xtrain = vectorizer.transform(x_train_array)
    Xtest = vectorizer.transform(x_test_array)
    Ytrain = data_train['category'].values
    Ytest = data_test['category'].values
    # LOGISTIC BASELINE DEFAULT PARAMETER
    classifier = LogisticRegression()
    # Create regularization penalty space

    # Create regularization hyperparameter distribution using uniform distribution

    params = dict(
        penalty=['l1', 'l2'],
        C=uniform(loc=0, scale=4))
    search = RandomizedSearchCV(classifier,
                                params, random_state=123456789, n_iter=200, cv=5, verbose=0, n_jobs=-1)

    search_result = search.fit(Xtrain, Ytrain)

    # Evaluate testing set using the best estimator
    test_accuracy = search.score(Xtest, Ytest)

    print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))

    means = search_result.cv_results_['mean_test_score']
    stds = search_result.cv_results_['std_test_score']
    params = search_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print(f'Best Accurcacy {search_result.best_score_}\n' +
          f'Best Param {search_result.best_params_}\n' +
          f'Test Accuracy {test_accuracy} ')


def main():
    run_nn: bool = True
    run_logisticbaseline: bool = False

    # LOAD DATA SET IN PANDAS
    datafilename_train = "News_category_train.json"
    datafilename_test = "News_category_test.json"
    path_to_file = "./datasets/"
    data_train = pd.read_json(os.path.join(path_to_file, datafilename_train))
    data_test = pd.read_json(os.path.join(path_to_file, datafilename_test))

    # CLEANING FROM THE TRAINING EXAMPLES IN TEST SET
    data_train = retain_unseen_example_in_test(data_train, data_test)

    if run_logisticbaseline:

        groups_of_features = [
            ['headline'],
            ['short_description'],
            ['authors'],
            ['headline', 'authors'],
            ['short_description', 'authors'],
            ['headline', 'short_description'],
            ['headline', 'short_description', 'authors']]

        for group in groups_of_features:
            evaluate_logistic(data_train,
                              data_test,
                              group)

    if run_nn:

        # EXTRACT TF-IDF BOW ARRAY
        columns = ['headline', 'short_description', 'authors']
        x_train_array = get_text_array(data_train, columns)
        x_test_array = get_text_array(data_test, columns)
        vectorizer = TfidfVectorizer(stop_words='english', analyzer='word',
                                     ngram_range=(1, 2), min_df=3, lowercase=True)
        vectorizer.fit(x_train_array)
        Xtrain = vectorizer.transform(x_train_array)
        Xtest = vectorizer.transform(x_test_array)

        # ONE HOT ENCODING OF CLASSES FOR DNN

        Ytrain = get_one_hot_class(data_train)
        Ytest = get_one_hot_class(data_test)
        # NEURAL NET 1 LAYER
        # DEFINE GRID SEARCH PARAMETERS NN. ARCHI
        param_grid = dict(dropout=[0.1, 0.2],
                          input_dim=[Xtrain.shape[1]],
                          output_dim=[Ytrain.shape[1]],
                          hidden_units=['10', '10_10', '100_100_100',
                                        '20', '20_20', '20_20_20',
                                        '100', '100_100', '100_100_100',
                                        '500', '500_500', '500_500_500'],
                          nb_epoch=[3, 4, 5],
                          batch_size=[32],
                          )

        model = KerasClassifier(build_fn=create_nn_model)
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=2,
                            n_jobs=-1,
                            verbose=0)
        grid_result = grid.fit(Xtrain, Ytrain)

        # Evaluate testing set using the best estimator
        test_accuracy = grid.score(Xtest, Ytest)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        print(f'Best Accurcacy {grid_result.best_score_}\n' +
              f'Best Param {grid_result.best_params_}\n' +
              f'Test Accuracy {test_accuracy} ')


if __name__ == '__main__':
    main()
