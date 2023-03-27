import os, json, random, mlflow
import pandas as pd
import numpy as np
from PIL import Image
from io import StringIO
from mlflow.pyfunc.scoring_server import predictions_to_json


# Set up MLflow tracking
experiment_name = "inference"+str(random.randint(10000,100000))
mlflow.set_experiment(experiment_name)
client = mlflow.MlflowClient()


def init(model_path):
    global model
    global input_schema
    # "model" is the path of the mlflow artifacts when the model was registered. For automl
    # models, this is generally "mlflow-model".
    model = mlflow.pyfunc.load_model(model_path)
    input_schema = model.metadata.get_input_schema()
    os.environ['MLFLOW_TRACKING_FORCE_NO_GIT'] = '1'
    os.environ['GIT_PYTHON_REFRESH'] = '0'

    
    
def parse_json_input(json_data):
    json_df = pd.read_json(json.dumps(json_data['dataframe_split']), orient='split')
    data = json_df.values.astype('float32')
    data = data.reshape(1, 28, 28, 1)
    return data


def average(lst):
    return sum(lst) / len(lst)


def run(raw_data):
    json_data = json.loads(raw_data)
    if 'dataframe_split' not in json_data.keys():
        raise Exception("Request must contain a top level key named 'dataframe_split'")

    data = parse_json_input(json_data)
    
    # Log the data as an artifact
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        Image.fromarray(data.reshape(28,28).astype(np.uint8)).save('data.png')
        mlflow.log_artifact("data.png")
    
    # Make predictions and log them
    with mlflow.start_run(run_id=run_id):
        predictions = model.predict(data)
        
        best_prediction = int(np.argmax(predictions, axis=1))
        best_proba = np.max(predictions, axis=1)
        worst_proba = np.min(predictions, axis=1)
        
        # log predicted probabilities as metrics
        for i, p in enumerate(predictions[0]):
            mlflow.log_metric(f'probability_class_{i}', p)
        
        # Log the predictions as an artifact
        with open("best_pred.txt", "w") as f:
            f.write(str(best_prediction))
        mlflow.log_artifact("best_pred.txt")

        # Log any anomalous predictions as a metric
        if best_proba < 0.9:
            mlflow.log_metric("anomalous_pred_proba", best_proba, step=i)
        if worst_proba > 0.1:
            mlflow.log_metric("anomalous_pred_proba", worst_proba, step=i)
        avg_preds = average(predictions[0])
        if round(avg_preds, 1) != 0.1:
            mlflow.log_metric('anomalous_avg_probas', avg_preds)
        sum_preds = sum(predictions[0])
        if round(sum_preds) != 1:
            mlflow.log_metric('anomalous_sum_probas', sum_preds)
        
    result = StringIO()
    predictions_to_json(best_prediction, result)
    return result.getvalue(), data
