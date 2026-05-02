import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"

def predict(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
            V11, V12, V13, V14, V15, V16, V17, V18,
            V19, V20, V21, V22, V23, V24, V25, V26,
            V27, V28, Amount):

    data = {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4,
        "V5": V5, "V6": V6, "V7": V7, "V8": V8,
        "V9": V9, "V10": V10, "V11": V11, "V12": V12,
        "V13": V13, "V14": V14, "V15": V15, "V16": V16,
        "V17": V17, "V18": V18, "V19": V19, "V20": V20,
        "V21": V21, "V22": V22, "V23": V23, "V24": V24,
        "V25": V25, "V26": V26, "V27": V27, "V28": V28,
        "Amount": Amount
    }

    response = requests.post(API_URL, json=data)
    result = response.json()

    return result

# Create UI
inputs = [gr.Number(label=f"V{i}") for i in range(1, 29)]
inputs.append(gr.Number(label="Amount"))

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="json",
    title="Fraud Detection System",
    description="Enter transaction features to predict fraud probability"
)

if __name__ == "__main__":
    interface.launch()