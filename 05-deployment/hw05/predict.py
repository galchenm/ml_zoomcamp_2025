import pickle

def load_model(model_path):
    """Load a machine learning model from a file.

    Args:
        model_path (str): The file path to the saved model.
    Returns:
        The loaded machine learning model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    model_path = 'pipeline_v1.bin'
    model = load_model(model_path)
    
    client = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    X = [client]
    pred = model.predict_proba(X)[0, 1]
    print(pred)