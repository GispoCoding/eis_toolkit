import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from eis_toolkit.feature_importance.feature_importance import evaluate_feature_importance

# here I set to paths
data_to_load = "PUT PATH TO X"
label_to_load = "PUT PATH TO Y"

if __name__ == "__main__":

    feature_names = [
        "Mag_TMI",
        "Mag_AS",
        "DRC135",
        "DRC180",
        "DRC45",
        "DRC90",
        "Mag_TD",
        "HDTDR",
        "Mag_Xdrv",
        "mag_Ydrv",
        "Mag_Zdrv",
        "Pseu_Grv",
        "Rd_U",
        "Rd_TC",
        "Rd_Th",
        "Rd_K",
        "EM_ratio",
        "EM_Ap_rs",
        "Em_Qd",
        "EM_Inph",
    ]

    # first things first let s load data
    X = pd.read_csv(f"{data_to_load}").to_numpy()
    y = pd.read_csv(f"{label_to_load}").to_numpy()

    # standardize the content
    X = StandardScaler().fit_transform(X)

    # now let s train a MLP classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # we can train a MLP classifier
    clf = MLPClassifier(solver="adam", alpha=0.001, hidden_layer_sizes=(16, 2), random_state=1)
    clf.fit(X_train, y_train.ravel())

    # we evaluate feature here
    evaluated_feature_importance, dictionary_of_features = evaluate_feature_importance(
        clf=clf, x_test=X_test, y_test=y_test, feature_names=feature_names, number_of_repetition=50, random_state=0
    )

    print(evaluated_feature_importance)

    # how to create a chart from here
    imp = pd.Series(dictionary_of_features.importances_mean * 100, index=feature_names).sort_values(ascending=True)
    ax = imp.plot.barh()
    ax.set_title("MLP Permutation Importance")
    ax.figure.tight_layout()
    plt.xlabel("Importance (%)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.ylabel("Feature")
    for i, v in enumerate(imp):
        ax.text(v, i, f"{v:.1f}", color="blue", fontweight="bold", fontsize=8)
    plt.savefig("testing.png")
