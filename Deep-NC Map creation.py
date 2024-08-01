import numpy as np
import rasterio
from rasterio.transform import from_origin
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def predict_fire_risk(model, input_rasters, output_path):
    # Đọc dữ liệu đầu vào
    raster_data = []
    for raster_path in input_rasters:
        with rasterio.open(raster_path) as src:
            raster_data.append(src.read(1))
            profile = src.profile
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    raster_data = np.array(raster_data).transpose(1, 2, 0)
    shape = raster_data.shape
    raster_data = scaler.fit_transform(raster_data.reshape(-1, shape[-1])).reshape(shape)
    
    # Dự đoán
    predictions = model.predict(raster_data)
    
    # Reshape kết quả về kích thước ban đầu
    risk_map = predictions.reshape(shape[0], shape[1])
    
    # Ghi kết quả ra file
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(risk_map.astype(rasterio.float32), 1)
    
    return risk_map

def classify_risk(risk_map, output_path, thresholds=[0.2, 0.4, 0.6, 0.8]):
    # Phân loại nguy cơ
    classified_risk = np.digitize(risk_map, thresholds)
    
    # Ghi kết quả phân loại ra file
    with rasterio.open(output_path, 'w', driver='GTiff', 
                       height=risk_map.shape[0], width=risk_map.shape[1],
                       count=1, dtype=rasterio.uint8, 
                       crs='+proj=latlong', transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(classified_risk.astype(rasterio.uint8), 1)
    
    return classified_risk

# Sử dụng các hàm

# Đường dẫn đến các raster đầu vào
input_rasters = [
    'slope.tif', 'aspect.tif', 'elevation.tif', 'curvature.tif',
    'ndvi.tif', 'ndwi.tif', 'ndmi.tif', 'landuse.tif',
    'temperature.tif', 'windspeed.tif', 'humidity.tif', 'rainfall.tif'
]

# Tải mô hình Deep-NC đã huấn luyện
model = tf.keras.models.load_model('deep_nc_model.h5')

# Dự đoán nguy cơ cháy rừng (FF - Forest Fire)
risk_map = predict_fire_risk(model, input_rasters, 'forest_fire_risk.tif')

# Phân loại và tích hợp kết quả vào GIS
classified_risk = classify_risk(risk_map, 'classified_forest_fire_risk.tif')

print("Đã hoàn thành dự đoán và phân loại nguy cơ cháy rừng.")