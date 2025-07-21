# dags/retrain_dag.py 2

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

default_args = {
    'owner': 'yash',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

def decide_branch():
    flag_path = '/opt/airflow/final_model/drift_detected.txt'
    try:
        with open(flag_path, 'r') as f:
            drift_flag = f.read().strip().lower()
    except FileNotFoundError:
        # If the file isn't there, assume no drift
        drift_flag = 'false'
    
    return 'retrain_model' if drift_flag == 'true' else 'skip_retrain'

with DAG(
    dag_id='model_retraining_on_drift',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    # Task 1: Run the drift detection script
    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='PYTHONPATH="/opt/airflow" python /opt/airflow/drift_detection/check_drift.py'
    )

    # Task 2: Branch based on drift flag
    branch_on_drift = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=decide_branch
    )

    # Task 3a: Retrain if drift detected
    retrain_model = BashOperator(
        task_id='retrain_model',
        bash_command='python /opt/airflow/model_training/train.py'
    )

    # Task 3b: Skip retraining if no drift
    skip_retrain = EmptyOperator(
        task_id='skip_retrain'
    )

    # Final end task
    end = EmptyOperator(
        task_id='end'
    )

    # Define dependencies
    detect_drift >> branch_on_drift
    branch_on_drift >> retrain_model >> end
    branch_on_drift >> skip_retrain >> end
