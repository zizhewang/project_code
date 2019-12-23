from PIL import Image
import numpy as np
import os
import scipy.io as sio

def generate_data_labels(path):
    filenames = os.listdir(path)
    #统计所有图片个数，初始化data和labels
    count = 0
    for filename in filenames:
        if filename == ".DS_Store":
            continue
        one_class_names = os.listdir(path + "/" + filename)
        for one_class_name in one_class_names:
            if one_class_name == ".DS_Store":
                continue
            try:
                img = np.array(Image.open(path + "/" + filename + "/" + one_class_name).resize([224, 224]))
            except:
                continue
            count += 1
    data = np.zeros([count, 224, 224, 3], dtype=np.uint8)
    labels = np.zeros([count], dtype=np.uint8)
    #读取图片
    meta = {}
    c = 0
    for idx, filename in enumerate(filenames):
        # 衣服种类对应一类(例：0:裙子，1:裤子...)
        meta[str(idx)] = filename
        one_class_names = os.listdir(path + "/" + filename)
        for one_class_name in one_class_names:
            if one_class_name == ".DS_Store":
                continue
            try:
                img = np.array(Image.open(path + "/" + filename + "/" + one_class_name).resize([224, 224]))
            except:
                continue
            if img.shape.__len__() < 3:
                img = np.dstack((img, img, img))
            data[c, :, :, :] = img
            labels[c] = idx
            c += 1
        print(filename)
    #.mat格式保存数据
    dataset = {}
    dataset["data"] = data
    dataset["labels"] = labels
    # dataset["metadata"] = meta
    sio.savemat("dataset.mat", dataset)
    sio.savemat("metadata.mat", meta)

if __name__ == "__main__":
    generate_data_labels("/Users/wangzizhe/Desktop/project/data/dataset")