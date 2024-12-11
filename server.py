from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from collections import Counter
from datetime import datetime
import os

app = Flask(__name__)

# Tải mô hình đã lưu
model = tf.keras.models.load_model('best_model2.keras')

# Đọc dữ liệu từ file CICFlowMeter


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Đọc cho trigger


def read_daily_flow_data():
    # Get today's date in the format yyyy-MM-dd
    today = datetime.now().strftime('%Y-%m-%d')

    # Construct the filename
    filename = f"{today}_Flow.csv"

    # Define the directory path
    directory = os.path.join('CICFlowMeter4', 'bin', 'data', 'daily')

    # Construct the full file path
    file_path = os.path.join(directory, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")

    # Read the last 500 lines of the file
    data = pd.read_csv(file_path).tail(140)

    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def get_data():
    data = read_daily_flow_data()
    test_X_clean, clean_indices = preprocess_data(data)
    test_predictions = predict_attack(test_X_clean)

    # Ensure the lengths of the arrays match
    if len(test_predictions) == len(clean_indices):
        data.loc[clean_indices, 'Predicted Label'] = test_predictions
    else:
        raise ValueError(
            "Shape mismatch: test_predictions and clean_indices do not match")

    # Count the occurrences of each label
    label_counts = Counter(data['Predicted Label'])
    predicted_labels = data['Predicted Label'].tolist()
    threshold = 0

    labels_above_threshold = [label for label,
                              count in label_counts.items() if count >= threshold]

    unique_labels = set()
    for label in labels_above_threshold:
        if pd.isna(label):
            unique_labels.add(-1)
        else:
            unique_labels.add(label)

    # If no labels meet the threshold, return 0
    if not unique_labels:
        response_label = 0
    else:
        response_label = list(unique_labels)
    checkres = [None if pd.isna(
        label) else label for label in predicted_labels]
    # Prepare the response
    response = {
        "response_label": response_label,
        "checkres": checkres
    }
    return jsonify(response)


def preprocess_data(data):
    required_features = [
        'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
        'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
        'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
        'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
        'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
    ]

    # Ánh xạ lại dữ liệu đã xử lý
    prepared_data = pd.DataFrame()
    prepared_data['Header_Length'] = data['Fwd Header Len'] + \
        data['Bwd Header Len']
    prepared_data['Protocol Type'] = data['Protocol']
    prepared_data['Duration'] = data['Flow Duration']
    prepared_data['Rate'] = data['Flow Byts/s']
    prepared_data['Srate'] = data['Flow Pkts/s']
    prepared_data['Drate'] = data['Flow Byts/s'] / \
        (data['Flow Duration'] + 1e-6)

    # Cờ mạng
    prepared_data['fin_flag_number'] = data['FIN Flag Cnt']
    prepared_data['syn_flag_number'] = data['SYN Flag Cnt']
    prepared_data['rst_flag_number'] = data['RST Flag Cnt']
    prepared_data['psh_flag_number'] = data['PSH Flag Cnt']
    prepared_data['ack_flag_number'] = data['ACK Flag Cnt']
    prepared_data['ece_flag_number'] = data['ECE Flag Cnt']
    prepared_data['cwr_flag_number'] = data['CWE Flag Count']

    prepared_data['ack_count'] = data['ACK Flag Cnt']
    prepared_data['syn_count'] = data['SYN Flag Cnt']
    prepared_data['fin_count'] = data['FIN Flag Cnt']
    prepared_data['rst_count'] = data['RST Flag Cnt']

    # Port mappings (HTTP, HTTPS, DNS, ...)
    protocol_ports = {
        'HTTP': [80],
        'HTTPS': [443],
        'DNS': [53],
        'Telnet': [23],
        'SMTP': [25],
        'SSH': [22],
        'IRC': [194]
    }
    for proto, ports in protocol_ports.items():
        prepared_data[proto] = data['Src Port'].isin(
            ports) | data['Dst Port'].isin(ports)

    prepared_data['TCP'] = data['Protocol'] == 6
    prepared_data['UDP'] = data['Protocol'] == 17
    prepared_data['DHCP'] = data['Src Port'].isin(
        [67, 68]) | data['Dst Port'].isin([67, 68])
    prepared_data['ARP'] = 0  # Không có dữ liệu ARP rõ ràng
    prepared_data['ICMP'] = data['Protocol'] == 1
    prepared_data['IGMP'] = data['Protocol'] == 2
    prepared_data['IPv'] = 0  # Không rõ từ dữ liệu gốc
    prepared_data['LLC'] = 0  # Không rõ từ dữ liệu gốc

    # Thống kê chiều dài gói
    prepared_data['Tot sum'] = data['Pkt Len Max'] + data['Pkt Len Min']
    prepared_data['Min'] = data['Pkt Len Min']
    prepared_data['Max'] = data['Pkt Len Max']
    prepared_data['AVG'] = data['Pkt Len Mean']
    prepared_data['Std'] = data['Pkt Len Std']
    prepared_data['Tot size'] = data['TotLen Fwd Pkts'] + \
        data['TotLen Bwd Pkts']
    prepared_data['IAT'] = data['Flow IAT Mean']
    prepared_data['Number'] = data['Tot Fwd Pkts'] + data['Tot Bwd Pkts']

    # Thêm các cột không có trong dữ liệu gốc với giá trị mặc định
    default_columns = ['Magnitue', 'Radius',
                       'Covariance', 'Variance', 'Weight']
    for col in default_columns:
        prepared_data[col] = 0

    # Mã hóa các cột phân loại (nếu có)
    encoder = LabelEncoder()
    if 'Protocol Type' in prepared_data.columns:
        prepared_data['Protocol Type'] = encoder.fit_transform(
            prepared_data['Protocol Type'])

    # Chọn các cột đặc trưng
    test_X = prepared_data[required_features]

    # Thay thế các giá trị vô hạn và giá trị lớn bằng NaN
    test_X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Loại bỏ các hàng có giá trị NaN và theo dõi các chỉ số
    test_X_clean = test_X.dropna()
    clean_indices = test_X_clean.index

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    test_X_clean = scaler.fit_transform(test_X_clean)

    print(f"Shape of test_X_clean: {test_X_clean.shape}")

    return test_X_clean, clean_indices

# Dự đoán với mô hình đã huấn luyện


def predict_attack(test_X_clean):
    test_predictions = model.predict(test_X_clean)
    if test_predictions.shape[1] > 1:
        # Assuming binary classification, select the first output
        test_predictions = np.argmax(test_predictions, axis=1)

    # Convert probabilities to binary predictions
    test_predictions = (test_predictions > 0.5).astype(int).flatten()

    # Handle NaN values in predictions
    test_predictions = np.nan_to_num(test_predictions, nan=0)

    return test_predictions

# Định nghĩa route cho API


@app.route('/predict', methods=['POST'])
def predict():

    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    print(file.filename)

    data = pd.read_csv(file).tail(500)
    test_X_clean, clean_indices = preprocess_data(data)
    test_predictions = predict_attack(test_X_clean)

    # Ensure the lengths of the arrays match
    if len(test_predictions) == len(clean_indices):
        data.loc[clean_indices, 'Predicted Label'] = test_predictions
    else:
        raise ValueError(
            "Shape mismatch: test_predictions and clean_indices do not match")

    # Count the occurrences of each label
    label_counts = Counter(data['Predicted Label'])
    threshold = 0

    labels_above_threshold = [label for label,
                              count in label_counts.items() if count >= threshold]

    unique_labels = set()
    for label in labels_above_threshold:
        if pd.isna(label):
            unique_labels.add(-1)
        else:
            unique_labels.add(label)

    # If no labels meet the threshold, return 0
    if not unique_labels:
        response_label = 0
    else:
        response_label = list(unique_labels)

    # Prepare the response
    response = {
        "response_label": response_label,
        "checkres": data['Predicted Label'].tolist()
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
