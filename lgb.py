import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
import xgboost
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt; plt.style.use('seaborn')



class Performance:

    def __init__(self, labels, scores, threshold=0.5):
        """
        :param labels:数组类型，真实的标签
        :param scores:数组类型，分类器的得分
        :param threshold:检测阈值
        """
        self.labels = labels
        self.scores = scores
        self.threshold = threshold
        self.db = self.get_db()
        self.TP, self.FP, self.FN, self.TN = self.get_confusion_matrix()

    def accuracy(self):
        """
        :return: 正确率
        """
        return 100*(self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    def presision(self):
        """
        :return: 准确率
        """
        return 100*self.TP / (self.TP + self.FP)

    def recall(self):
        """
        :return: 召回率
        """
        return 100*self.TP / (self.TP + self.FN)

    def auc(self):
        """
        :return: auc值
        """
        auc = 0.
        prev_x = 0
        xy_arr = self.roc_coord()
        for x, y in xy_arr:
            if x != prev_x:
                auc += (x - prev_x) * y
                prev_x = x
        return auc

    def roc_coord(self):
        """
        :return: roc坐标
        """
        xy_arr = []
        tp, fp = 0., 0.
        neg = self.TN + self.FP
        pos = self.TP + self.FN
        for i in range(len(self.db)):
            tp += self.db[i][0]
            fp += 1 - self.db[i][0]
            xy_arr.append([fp / neg, tp / pos])
        return xy_arr

    def roc_plot(self):
        """
        画roc曲线
        :return:
        """
        auc = self.auc()
        xy_arr = self.roc_coord()
        x = [_v[0] for _v in xy_arr]
        y = [_v[1] for _v in xy_arr]
        plt.title("ROC curve (AUC = %.4f)" % auc)
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.plot(x, y)
        plt.show()

    def get_db(self):
        db = []
        for i in range(len(self.labels)):
            db.append([self.labels[i], self.scores[i]])
        db = sorted(db, key=lambda x: x[1], reverse=True)
        return db

    def get_confusion_matrix(self):
        """
        计算混淆矩阵
        :return:
        """
        tp, fp, fn, tn = 0., 0., 0., 0.
        for i in range(len(self.labels)):
            if self.labels[i] == 1 and self.scores[i] >= self.threshold:
                tp += 1
            elif self.labels[i] == 0 and self.scores[i] >= self.threshold:
                fp += 1
            elif self.labels[i] == 1 and self.scores[i] < self.threshold:
                fn += 1
            else:
                tn += 1
        return [tp, fp, fn, tn]

# 画出损失函数的变化情况
def plot_logloss(model):
    results = model.evals_result_
    print(results)
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    print(epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    print(results['validation_1']['logloss'])
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGDboost Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoostClassification Error')
    plt.show()

data = pd.read_csv('./result_csv/feature-all.csv')
name = ['GLCM_ASM_0', 'GLCM_Contrast_0', 'GLCM_Correlation_0', 'GLCM_SumOfSquaresVariance_0',
        'GLCM_InverseDifferenceMoment_0', 'GLCM_SumAverage_0', 'GLCM_SumVariance_0', 'GLCM_SumEntropy_0',
        'GLCM_Entropy_0', 'GLCM_DifferenceVariance_0', 'GLCM_DifferenceEntropy_0', 'GLCM_Information1_0',
        'GLCM_Information2_0', 'GLCM_MaximalCorrelationCoefficient_0', 'GLCM_ASM_45', 'GLCM_Contrast_45', 'GLCM_Correlation_45', 'GLCM_SumOfSquaresVariance_45', 'GLCM_InverseDifferenceMoment_45', 'GLCM_SumAverage_45', 'GLCM_SumVariance_45', 'GLCM_SumEntropy_45', 'GLCM_Entropy_45', 'GLCM_DifferenceVariance_45', 'GLCM_DifferenceEntropy_45', 'GLCM_Information1_45', 'GLCM_Information2_45', 'GLCM_MaximalCorrelationCoefficient_45', 'GLCM_ASM_90', 'GLCM_Contrast_90', 'GLCM_Correlation_90', 'GLCM_SumOfSquaresVariance_90', 'GLCM_InverseDifferenceMoment_90', 'GLCM_SumAverage_90', 'GLCM_SumVariance_90', 'GLCM_SumEntropy_90', 'GLCM_Entropy_90', 'GLCM_DifferenceVariance_90', 'GLCM_DifferenceEntropy_90', 'GLCM_Information1_90', 'GLCM_Information2_90', 'GLCM_MaximalCorrelationCoefficient_90', 'GLCM_ASM_135', 'GLCM_Contrast_135', 'GLCM_Correlation_135', 'GLCM_SumOfSquaresVariance_135', 'GLCM_InverseDifferenceMoment_135', 'GLCM_SumAverage_135', 'GLCM_SumVariance_135', 'GLCM_SumEntropy_135', 'GLCM_Entropy_135', 'GLCM_DifferenceVariance_135', 'GLCM_DifferenceEntropy_135', 'GLCM_Information1_135', 'GLCM_Information2_135', 'GLCM_MaximalCorrelationCoefficient_135', 'GLCM_ASM_Mean', 'GLCM_Contrast_Mean', 'GLCM_Correlation_Mean', 'GLCM_SumOfSquaresVariance_Mean', 'GLCM_InverseDifferenceMoment_Mean', 'GLCM_SumAverage_Mean', 'GLCM_SumVariance_Mean', 'GLCM_SumEntropy_Mean', 'GLCM_Entropy_Mean', 'GLCM_DifferenceVariance_Mean', 'GLCM_DifferenceEntropy_Mean', 'GLCM_Information1_Mean', 'GLCM_Information2_Mean', 'GLCM_MaximalCorrelationCoefficient_Mean', 'GLCM_ASM_Range', 'GLCM_Contrast_Range', 'GLCM_Correlation_Range', 'GLCM_SumOfSquaresVariance_Range', 'GLCM_InverseDifferenceMoment_Range', 'GLCM_SumAverage_Range', 'GLCM_SumVariance_Range', 'GLCM_SumEntropy_Range', 'GLCM_Entropy_Range', 'GLCM_DifferenceVariance_Range', 'GLCM_DifferenceEntropy_Range', 'GLCM_Information1_Range', 'GLCM_Information2_Range', 'GLCM_MaximalCorrelationCoefficient_Range', 'GLCM_ASM_Mean.1', 'GLCM_Contrast_Mean.1', 'GLCM_Correlation_Mean.1', 'GLCM_SumOfSquaresVariance_Mean.1', 'GLCM_InverseDifferenceMoment_Mean.1', 'GLCM_SumAverage_Mean.1', 'GLCM_SumVariance_Mean.1', 'GLCM_SumEntropy_Mean.1', 'GLCM_Entropy_Mean.1', 'GLCM_DifferenceVariance_Mean.1', 'GLCM_DifferenceEntropy_Mean.1', 'GLCM_Information1_Mean.1', 'GLCM_Information2_Mean.1', 'GLCM_MaximalCorrelationCoefficient_Mean.1', 'GLCM_ASM_Range.1', 'GLCM_Contrast_Range.1', 'GLCM_Correlation_Range.1', 'GLCM_SumOfSquaresVariance_Range.1', 'GLCM_InverseDifferenceMoment_Range.1', 'GLCM_SumAverage_Range.1', 'GLCM_SumVariance_Range.1', 'GLCM_SumEntropy_Range.1', 'GLCM_Entropy_Range.1', 'GLCM_DifferenceVariance_Range.1', 'GLCM_DifferenceEntropy_Range.1', 'GLCM_Information1_Range.1', 'GLCM_Information2_Range.1', 'GLCM_MaximalCorrelationCoefficient_Range.1', 'GLSZM_SmallZoneEmphasis', 'GLSZM_LargeZoneEmphasis', 'GLSZM_GrayLevelNonuniformity', 'GLSZM_ZoneSizeNonuniformity', 'GLSZM_ZonePercentage', 'GLSZM_LowGrayLeveLZoneEmphasis', 'GLSZM_HighGrayLevelZoneEmphasis', 'GLSZM_SmallZoneLowGrayLevelEmphasis', 'GLSZM_SmallZoneHighGrayLevelEmphasis', 'GLSZM_LargeZoneLowGrayLevelEmphassis', 'GLSZM_LargeZoneHighGrayLevelEmphasis', 'GLSZM_GrayLevelVariance', 'GLSZM_ZoneSizeVariance', 'GLSZM_ZoneSizeEntropy', 'NGTDM_Coarseness', 'NGTDM_Contrast', 'NGTDM_Busyness', 'NGTDM_Complexity', 'NGTDM_Strngth', 'LBP_R8_P1_0', 'LBP_R8_P1_1', 'LBP_R8_P1_2', 'LBP_R8_P1_3', 'LBP_R8_P1_4', 'LBP_R8_P1_5', 'LBP_R8_P1_6', 'LBP_R8_P1_7', 'LBP_R8_P1_8', 'LBP_R8_P1_9', 'LBP_R8_P1_10', 'LBP_R8_P1_11', 'LBP_R8_P1_12', 'LBP_R8_P1_13', 'LBP_R8_P1_14', 'LBP_R8_P1_15', 'LBP_R8_P1_16', 'LBP_R8_P1_17', 'LBP_R8_P1_18', 'LBP_R8_P1_19', 'LBP_R8_P1_20', 'LBP_R8_P1_21', 'LBP_R8_P1_22', 'LBP_R8_P1_23', 'LBP_R8_P1_24', 'LBP_R8_P1_25', 'LBP_R8_P1_26', 'LBP_R8_P1_27', 'LBP_R8_P1_28', 'LBP_R8_P1_29', 'LBP_R8_P1_30', 'LBP_R8_P1_31', 'LBP_R8_P1_32', 'LBP_R8_P1_33', 'LBP_R8_P1_34', 'LBP_R8_P1_35', 'LBP_R8_P1_36', 'LBP_R8_P1_37', 'LBP_R8_P1_38', 'LBP_R8_P1_39', 'LBP_R8_P1_40', 'LBP_R8_P1_41', 'LBP_R8_P1_42', 'LBP_R8_P1_43', 'LBP_R8_P1_44', 'LBP_R8_P1_45', 'LBP_R8_P1_46', 'LBP_R8_P1_47', 'LBP_R8_P1_48', 'LBP_R8_P1_49', 'LBP_R8_P1_50', 'LBP_R8_P1_51', 'LBP_R8_P1_52', 'LBP_R8_P1_53', 'LBP_R8_P1_54', 'LBP_R8_P1_55', 'LBP_R8_P1_56', 'LBP_R8_P1_57', 'LBP_R8_P1_58']

X_train, X_test, Y_train, Y_test = train_test_split(data[name], data['label'], test_size=0.3, random_state=3)

# 模型训练
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=200)

gbm.fit(X_train, Y_train,
        eval_set=[(X_test, Y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

result = np.int64(y_pred>0.5)
test_y = Y_test.reset_index(drop=True)

p = Performance(test_y, result)
acc = p.accuracy()
pre = p.presision()
rec = p.recall()

print('accuracy: %.2f %%' % acc)
print('precision: %.2f %%' % pre)
print('recall: %.2f %%' % rec)
p.roc_plot()


