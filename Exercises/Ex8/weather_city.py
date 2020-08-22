import sys
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC


def main(labelled, unlabelled, output_fn):
    # Train Model
    X = labelled.loc[:, 'tmax-01':'snwd-12'].values
    y = labelled['city']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), SVC(kernel='linear', C=0.1))
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))

    # Make prediction
    data = unlabelled.loc[:, 'tmax-01':'snwd-12'].values
    predictions = model.predict(data)
    pd.Series(predictions).to_csv(output_fn, index=False, header=False)


if __name__ == '__main__':
    labelled = pd.read_csv(sys.argv[1])
    unlabelled = pd.read_csv(sys.argv[2])
    output_fn = sys.argv[3]
    main(labelled, unlabelled, output_fn)
