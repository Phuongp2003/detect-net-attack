from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

app = Flask(__name__)

# Tải mô hình đã lưu
model = tf.keras.models.load_model('best_model.keras')


# Đọc dữ liệu từ file test_data.csv
test_data = pd.read_csv('test_data.csv')

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

# Tính toán và ánh xạ dữ liệu vào các cột
prepared_data['Header_Length'] = test_data['Fwd Header Len'] + \
    test_data['Bwd Header Len']
prepared_data['Protocol Type'] = test_data['Protocol']
prepared_data['Duration'] = test_data['Flow Duration']
prepared_data['Rate'] = test_data['Flow Byts/s']
prepared_data['Srate'] = test_data['Flow Pkts/s']
prepared_data['Drate'] = test_data['Flow Byts/s'] / \
    (test_data['Flow Duration'] + 1e-6)

# Cờ mạng
prepared_data['fin_flag_number'] = test_data['FIN Flag Cnt']
prepared_data['syn_flag_number'] = test_data['SYN Flag Cnt']
prepared_data['rst_flag_number'] = test_data['RST Flag Cnt']
prepared_data['psh_flag_number'] = test_data['PSH Flag Cnt']
prepared_data['ack_flag_number'] = test_data['ACK Flag Cnt']
prepared_data['ece_flag_number'] = test_data['ECE Flag Cnt']
prepared_data['cwr_flag_number'] = test_data['CWE Flag Count']

prepared_data['ack_count'] = test_data['ACK Flag Cnt']
prepared_data['syn_count'] = test_data['SYN Flag Cnt']
prepared_data['fin_count'] = test_data['FIN Flag Cnt']
prepared_data['rst_count'] = test_data['RST Flag Cnt']

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
    prepared_data[proto] = test_data['Src Port'].isin(
        ports) | test_data['Dst Port'].isin(ports)

prepared_data['TCP'] = test_data['Protocol'] == 6
prepared_data['UDP'] = test_data['Protocol'] == 17
prepared_data['DHCP'] = test_data['Src Port'].isin(
    [67, 68]) | test_data['Dst Port'].isin([67, 68])
prepared_data['ARP'] = 0  # Không có dữ liệu ARP rõ ràng
prepared_data['ICMP'] = test_data['Protocol'] == 1
prepared_data['IGMP'] = test_data['Protocol'] == 2
prepared_data['IPv'] = 0  # Không rõ từ dữ liệu gốc
prepared_data['LLC'] = 0  # Không rõ từ dữ liệu gốc

# Thống kê chiều dài gói
prepared_data['Tot sum'] = test_data['Pkt Len Max'] + test_data['Pkt Len Min']
prepared_data['Min'] = test_data['Pkt Len Min']
prepared_data['Max'] = test_data['Pkt Len Max']
prepared_data['AVG'] = test_data['Pkt Len Mean']
prepared_data['Std'] = test_data['Pkt Len Std']
prepared_data['Tot size'] = test_data['TotLen Fwd Pkts'] + \
    test_data['TotLen Bwd Pkts']
prepared_data['IAT'] = test_data['Flow IAT Mean']
prepared_data['Number'] = test_data['Tot Fwd Pkts'] + test_data['Tot Bwd Pkts']

# Thêm các cột không có trong dữ liệu gốc với giá trị mặc định
default_columns = ['Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
for col in default_columns:
    prepared_data[col] = 0

# Kiểm tra các cột trong prepared_data
print(prepared_data.columns)

# Lưu dữ liệu ra file CSV
prepared_file_path = 'prepared_test_data.csv'
prepared_data.to_csv(prepared_file_path, index=False)

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

# Dự đoán với mô hình đã huấn luyện
test_predictions = model.predict(test_X_clean)
test_predictions = (test_predictions > 0.5).astype(
    int).flatten()  # Đảm bảo là mảng 1D

# Kiểm tra kích thước của các mảng
print(f"test_predictions shape: {test_predictions.shape}")
print(f"clean_indices shape: {clean_indices.shape}")

# Đảm bảo kích thước của các mảng khớp nhau
if len(test_predictions) == len(clean_indices):
    prepared_data.loc[clean_indices, 'Predicted Label'] = test_predictions
else:
    print(f"test_predictions: {test_predictions}")
    print(f"clean_indices: {clean_indices}")
    raise ValueError(
        "Shape mismatch: test_predictions and clean_indices do not match")


if __name__ == '__main__':
    app.run(debug=True)
