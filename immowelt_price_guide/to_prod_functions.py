import mlflow
import pickle
import shutil
import yaml
import re

mlflow.set_tracking_uri("http://localhost:5000")

def set_model_to_prod(model_name, path="backend/prod_model.pkl"):
    model_pickle = load_model(model_name)
    with open(path, 'wb') as file:
         model_pickle = pickle.dump(model_pickle, file)

def set_model_requirements_to_prod(original_path, destination_path="backend/requirements.txt"):
    # Additional packages to add
    additional_packages = ["fastapi\n", "pandas\n"]
    with open(original_path, "r") as file:
        original_content = file.readlines()
    modified_content = additional_packages + original_content
    with open(destination_path, "w") as file:
        file.writelines(modified_content)

def get_model_details_path(model_name):
    source_path = get_model_source_path(model_name)
    return source_path + "MLmodel"

def get_model_req_path(model_name):
    source_path = get_model_source_path(model_name)
    return source_path + "requirements.txt"

def get_model_env_path(model_name):
    source_path = get_model_source_path(model_name)
    return source_path + "python_env.yaml"

def update_dockerfile_with_python_version(yaml_path, dockerfile_path="backend/Dockerfile"):
    # Load the YAML file
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Extract the Python version
    python_version = yaml_data.get('python', '')

    # Extract the major and minor version components
    major, minor, _ = python_version.split('.')
    python_version = f"{major}.{minor}"

    # Update the Dockerfile
    with open(dockerfile_path, 'r') as file:
        dockerfile = file.read()

    pattern = r'FROM tiangolo/uvicorn-gunicorn-fastapi:python\d+\.\d+'
    updated_dockerfile = re.sub(pattern,
                                f'FROM tiangolo/uvicorn-gunicorn-fastapi:python{python_version}',
                                dockerfile)

    with open(dockerfile_path, 'w') as file:
        file.write(updated_dockerfile)


def set_model_details_to_prod(source_path, destination_path="backend/prod_model_details.txt"):
    shutil.copy(source_path, destination_path)

def load_model(model_name, stage="production"):
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    return model

def get_model_source_path(model_name):
    client = mlflow.tracking.MlflowClient()
    model_details = client.get_registered_model(model_name)
    source = model_details.latest_versions[0].source
    return "mlartifacts" + source[source.index("/"):] + "/"