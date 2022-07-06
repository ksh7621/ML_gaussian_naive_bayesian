import numpy as np
import math
import cv2
import glob
import os


def data_loader(path):
    img_path = glob.glob(path) 
    feature_info = list()
    label_info = list()
    
    for i in range(len(img_path)):
#         print(img_path[i])
        label_info.append((img_path[i].split('-')[-1]).split('.')[0])
        img = cv2.imread(img_path[i], cv2.IMREAD_GRAYSCALE)
        feature_info.append(img.reshape(1,-1))
    #     print(img.shape)
    #     print(type(img))
    #     print(img)
    
    return feature_info, label_info

train_set_path = './train/*.jpg'
test_set_path = './test/*.jpg'

train_samples, train_labels = data_loader(train_set_path)
test_samples, test_labels = data_loader(test_set_path)


def min_max_norm(train_samples):
    
    data_list = np.squeeze(np.asarray(train_samples))
    normalized_data = []
    for data in data_list:
        normalized_x = (data-min(data))/(max(data)-min(data))
        normalized_data.append(np.expand_dims(normalized_x, axis = 0))
    return normalized_data

train_samples = min_max_norm(train_samples)
test_samples = min_max_norm(test_samples)


def get_GaussianNBC(train_samples, train_labels):
    fashion_class_samples0 = []
    fashion_class_samples1 = []
    fashion_class_samples2 = []
    fashion_class_samples3 = []
    fashion_class_samples4 = []
    fashion_class_samples5 = []
    fashion_class_samples6 = []
    fashion_class_samples7 = []
    fashion_class_samples8 = []
    fashion_class_samples9 = [] 
    
    for k in range(len(train_samples)):
        sample = train_samples[k]
        label = train_labels[k]
        
        if label == '0':
            fashion_class_samples0.append(sample)
        elif label == '1':
            fashion_class_samples1.append(sample)
        elif label == '2':
            fashion_class_samples2.append(sample)
        elif label == '3':
            fashion_class_samples3.append(sample)
        elif label == '4':
            fashion_class_samples4.append(sample)
        elif label == '5':
            fashion_class_samples5.append(sample)
        elif label == '6':
            fashion_class_samples6.append(sample)
        elif label == '7':
            fashion_class_samples7.append(sample)
        elif label == '8':
            fashion_class_samples8.append(sample)
        elif label == '9':
            fashion_class_samples9.append(sample)
            
    
    samples_by_classes = [
        fashion_class_samples0,
        fashion_class_samples1,
        fashion_class_samples2,
        fashion_class_samples3,
        fashion_class_samples4,
        fashion_class_samples5,
        fashion_class_samples6,
        fashion_class_samples7,
        fashion_class_samples8,
        fashion_class_samples9
    ]    
    
    
    numOf_classes = 10
    means_by_classes = []
    stdev_by_classes = []

    for C in range(numOf_classes):
        # print('C', C)
        means = []
        stdevs = []
        # # for features in zip(*samples_by_classes[C]):
        # for i in range(len(samples_by_classes[C])):
        #     features = samples_by_classes[C][i]
        #     print(cnt+1)
        
        features = np.squeeze(np.asarray(samples_by_classes[C]))  # shape: [N_c, D]
        means.append(np.mean(features, axis=0))  # [784]
        stdevs.append(np.std(features, axis=0))  # [784]
            
            
        means_by_classes.append(means)  # shape: [10,784]
        stdev_by_classes.append(stdevs)  # shape: [10,784]
        
    return means_by_classes, stdev_by_classes

means_by_classes, stdev_by_classes = get_GaussianNBC(train_samples,train_labels)

def Gaussian_PDF(x, mean, stdev):
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    return (1/(math.sqrt(2*math.pi)*stdev)) *\
        (math.exp(-(math.pow(x-mean,2) / (2*math.pow(stdev,2)))))


def predict(means, stdevs, test_samples):
    pred_classes = []
    numOf_classes = 10
    numOf_features = 28
    
    for i in range(len(test_samples)):
        prob_by_classes = []
        for C in range(numOf_classes):
            prob = 1
            for j in range(numOf_features):
                
                mean = means[C][0][j]
                stdev = stdevs[C][0][j]
                x =  test_samples[i][0][j]
                # print(x.shape, stdev.shape, mean.shape)
                prob *= Gaussian_PDF(x, mean, stdev)
            prob_by_classes.append(prob)
        
        bestProb = -1
        for C in range(numOf_classes):
            if prob_by_classes[C] > bestProb:
                bestProb = prob_by_classes[C]
                pred_Label = C
        pred_classes.append(pred_Label)
    return pred_classes

pred_classes = predict(means_by_classes, stdev_by_classes, test_samples)


def get_Acc(pred_classes, test_labels):
    a = np.asarray(pred_classes)
    test_labels = [int(x) for x in test_labels]
    b = np.asarray(test_labels)
    accuracy = a == b
    # accuracy = np.equal(pred_classes_of_testset, gt_of_testset)
    return list(accuracy).count(True) / len(accuracy) * 100
acc = get_Acc(pred_classes, test_labels)
print('Acc: %s' % acc)


