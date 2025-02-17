import logging
import mlflow
import yaml
from sklearn.metrics import classification_report

from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("First experiment")
    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Clean data
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Prepare and train model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")

        # Evaluate model
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, trainer.pipeline.predict(X_test), output_dict=True)
        logging.info("Model evaluation completed successfully")

        # set tags
        mlflow.set_tag("Model developer", "prsdm")
        mlflow.set_tag('preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')

        #log metrics
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc", roc_auc_score)
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        mlflow.sklearn.log_model(trainer.pipeline, "model")

        # register model
        model_name = "insurance_model"
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        logging.info("Model registered successfully")

        # Print evaluation results
        print("n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"n{class_report}")
        print("=====================================================n")

if __name__ == "__main__":
    main()