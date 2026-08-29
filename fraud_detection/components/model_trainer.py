import os
import sys
from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from fraud_detection.entity.config_entity import ModelTrainerConfig
from fraud_detection.utils.main_utils.utils import save_object, load_numpy_array_data, load_object, evaluate_models
from fraud_detection.utils.ml_utils.metric.classification_metric import get_classification_score
from fraud_detection.utils.ml_utils.model.estimator import FraudDetectionModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, model_trainer_config:ModelTrainerConfig):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys)


    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
                    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
                    "Random Forest": RandomForestClassifier(class_weight="balanced", n_jobs=-1),
                    "XGBoost": XGBClassifier(scale_pos_weight=24, n_jobs=-1),
                }

        params = {
                    "Logistic Regression": {
                        'C': [0.01, 0.1, 1.0, 10.0],  
                        'solver': ['lbfgs', 'liblinear']
                    },
                    "Decision Tree": {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10] 
                    },
                    "Random Forest": {
                        'n_estimators': [128, 256],  
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    },
                    "XGBoost": {
                        'learning_rate': [0.1, 0.05, 0.01], 
                        'max_depth': [3, 5, 7, 9],          
                        'n_estimators': [128, 256],
                        'subsample': [0.8, 1.0] 
                    }
            }

        model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models, params)

        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        logging.info(f"Best Model Found: {best_model_name} with score: {best_model_score}")

        # Safety Check: If the best model is still garbage, fail the pipeline
        if best_model_score < self.model_trainer_config.expected_accuracy:
            raise Exception("No best model found. All models performed below the expected accuracy threshold.")

        
        y_train_pred=best_model.predict(X_train)

        classification_train_metric= get_classification_score(y_train, y_train_pred)

        y_test_pred=best_model.predict(X_test)
        classification_test_metric=get_classification_score(y_test, y_test_pred)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        fraud_model_obj = FraudDetectionModel(preprocessor, best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=fraud_model_obj)

        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact = classification_train_metric,
                             test_metric_artifact = classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys)