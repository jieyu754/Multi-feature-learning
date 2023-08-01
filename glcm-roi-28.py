import pyfeats
import os
import numpy as np
from PIL import Image
import mahotas
import pandas as pd
import cv2

def glcm_features(f, ignore_zeros=True):

    f = f.astype(np.uint8)

    # 3) Calculate Features: Mean and Range

    # return_mean： 布尔， 可选
    # 设置后，该函数返回所有方向的平均值（默认值：False）。
    # return_mean_ptp： 布尔， 可选   #如果设置为  true 输出为（28，） False则输出 4x14
    # 设置后，该函数返回所有方向上的平均值和 ptp（点对点，即 max（） 和 min（）之间的差值（默认值：False））

    features = mahotas.features.haralick(f,
                                         ignore_zeros=ignore_zeros,
                                         compute_14th_feature=True,
                                         return_mean_ptp=True,
                                         )
    features_mean = features[0:14]
    features_range = features[14:]

    return features



if __name__ == '__main__':
    values = []
    # 读取images文件夹下所有文件的名字
    img_path = r'C:\Users\jieyu\Desktop\SA-COVID-main\images'
    mask_path2 = r'C:\Users\jieyu\Desktop\SA-COVID-main\mask23'
    df = pd.read_csv("featues.csv")
    image_names = df['name']

    for i in range(len(image_names)):

        image_path = os.path.join(img_path, image_names[i])
        mask_path = os.path.join(mask_path2, image_names[i])
        img = np.array(Image.open(image_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = mask / 255
        masked = img * mask
        features = glcm_features(masked)
        kk = np.array(features).flatten()
        values.append(kk)

    names = ["GLCM_ASM", "GLCM_Contrast", "GLCM_Correlation",
             "GLCM_SumOfSquaresVariance", "GLCM_InverseDifferenceMoment",
             "GLCM_SumAverage", "GLCM_SumVariance", "GLCM_SumEntropy",
             "GLCM_Entropy", "GLCM_DifferenceVariance",
             "GLCM_DifferenceEntropy", "GLCM_Information1",
             "GLCM_Information2", "GLCM_MaximalCorrelationCoefficient"]

    name_mean = [name + "_Mean" for name in names]
    name_range = [name + "_Range" for name in names]
    features_all_name = name_mean + name_range



    test = pd.DataFrame(columns=features_all_name, index=image_names, data=values)
    test.to_csv('./glcm-roi-28-inf.csv', index=True)