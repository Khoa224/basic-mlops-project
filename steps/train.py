import os.path
import joblib
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config["model"]["name"]
        self.model_params = self.config["model"]["params"]
        self.model_path = self.config["model"]["store_path"]
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)

    def create_pipeline(self):
        # preprocessing pipeline for each data type
        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standardize', StandardScaler(), ['Age', 'RegionID']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'PastAccident']),
        ])
        # smote for imbalanced data handling
        smote = SMOTE(sampling_strategy=1)

        # map model name to class
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }

        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])

        return pipeline

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def train_model(self, X_train,y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        model_path = os.path.join(self.model_path, 'model.pkl')

        joblib.dump(self.pipeline, model_path)
