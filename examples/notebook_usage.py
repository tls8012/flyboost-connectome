from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flyboost import FlyBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

for mode in ["real", "shuffled", "frozen"]:
    print(f"\n=== {mode} ===")
    model = FlyBoostClassifier(
        graph_path="data/malecns_nomotor.pt",
        n_flies=8,
        learning_rate=0.2,
        fly_steps=8,
        epochs_per_fly=8,
        batch_size=32,
        fly_lr=2e-3,
        fly_mode=mode,
        device="cpu",
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    print("score:", model.score(X_val, y_val))
