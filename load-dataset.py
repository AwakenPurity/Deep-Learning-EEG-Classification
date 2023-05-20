import pdb

import numpy as np
import glob
import mne
import pywt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import Counter

file_name = 'D:/dataset_wj/sleep-edf-database-expanded-1.0.0/test/'
psg_fnames = glob.glob(os.path.join(file_name, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(file_name, "*Hypnogram.edf"))
psg_fnames.sort()
ann_fnames.sort()
record_annotation = tmp = list(zip(psg_fnames, ann_fnames))

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def loadData():
    dataSet = []
    labelSet = []
    i = 0
    for psg,ann in record_annotation:
        i = i+1
        print(f'正在加载第{i}个文件.......')
        print('-' * 30)
        mapping = {'EOG horizontal': 'eog',
                   'Resp oro-nasal': 'misc',
                   'EMG submental': 'misc',
                   'Temp rectal': 'misc',
                   'Event marker': 'misc'}
        # d读取 PSG文件 和 注释文件
        raw_train = mne.io.read_raw_edf(psg, preload=False)
        annot_train = mne.read_annotations(ann)
        raw_train.set_annotations(annot_train, emit_warning=False)
        raw_train.set_channel_types(mapping)

        '''
        一开始的时间列表为event_dict: 
                {'Sleep stage 1': 1, 
                'Sleep stage 2': 2, 
                'Sleep stage 3': 3, 
                'Sleep stage 4': 4, 
                'Sleep stage ?': 5, 
                'Sleep stage R': 6, 
                'Sleep stage W': 7
                }
        但我们需要对其进行重新映射，改成如下：
        '''
        annotation_change = {'Sleep stage W': 0,
                                      'Sleep stage 1': 1,
                                      'Sleep stage 2': 2,
                                      'Sleep stage 3': 3,
                                      'Sleep stage 4': 3,
                                      'Sleep stage R': 4}
        events_train, event_train_dict = mne.events_from_annotations(
            raw_train, event_id=annotation_change, chunk_duration=30.)

        # 创建一个新的event_id以统一 阶段3和4
        event_id = {'Sleep stage W': 0,
                    'Sleep stage 1': 1,
                    'Sleep stage 2': 2,
                    'Sleep stage 3/4': 3,
                    'Sleep stage R': 4}
        tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
        try:
            epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                                  event_id=event_id, tmin=0., tmax=tmax, baseline=None)
        except:
            print(f'{psg} No matching events found for Sleep stage !!! ')
            continue

        epochs_train.load_data()
        # 获取通道'EEG Fpz-Cz'的数据集
        # 存储数据
        data = epochs_train.pick_channels(['EEG Fpz-Cz'])
        for item in zip(data):
            item = np.array(item).flatten()
            dataSet.append(list(item))
        # 存储标签
        label = epochs_train.events[:, 2]
        for item in label:
            labelSet.append(item)
        print('-'*30)

    dataSet = np.array(dataSet) # (364059, 3000), <class 'numpy.ndarray'>
    labelSet = np.array(labelSet)

    # 数据打混，分测试集和验证集
    dataSet = dataSet.reshape(-1,3000)
    labelSet = labelSet.reshape(-1,1)
    total_data = np.hstack((dataSet, labelSet))
    np.random.shuffle(total_data)

    X = total_data[:,:3000].reshape(-1,3000,1)
    Y = total_data[:, 3000]

    RATIO = 0.3
    shuffle_index = np.random.permutation(len(total_data))
    test_length = int(0.3*len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]

    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]

    return X_train, Y_train, X_test, Y_test

'''
X_test.shape: (109217, 3000), 
X_test[0].shape: (3000,)
Y_test：(109217,)

train_Data: Counter({1.0: 175257, 3.0: 42339, 5.0: 15778, 2.0: 12440, 4.0: 9028})
'''
import torch
import torch.utils.data as Data
from torch import nn

X_train, Y_train, X_test, Y_test = loadData()
print(type(X_train))
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))
print(X_train.shape)
print(X_train[0])
print(Y_train.shape)

train_Data = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train)) # 返回结果为一个个元组，每一个元组存放数据和标签
train_loader = Data.DataLoader(dataset=train_Data, batch_size=128)
test_Data = Data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test)) # 返回结果为一个个元组，每一个元组存放数据和标签
test_loader = Data.DataLoader(dataset=test_Data, batch_size=128)


class RnnModel(nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()
        '''
        参数解释：(输入维度，隐藏层维度，网络层数)
        输入维度：每个x的输入大小，也就是每个x的特征数
        隐藏层：隐藏层的层数，若层数为1，隐层只有1层
        网络层数：网络层的大小
        '''
        self.rnn = nn.RNN(3000, 50, 3, nonlinearity='tanh')
        self.linear = nn.Linear(50, 5)

    def forward(self, x):
        r_out, h_state = self.rnn(x)
        output = self.linear(r_out[:,-1,:])
        return output

model = RnnModel()

'''
设置损失函数和参数优化方法
'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

'''
模型训练
'''
result = 0
EPOCHS = 5
for epoch in range(EPOCHS):
    running_loss = 0
    for i, data in enumerate(train_loader):
        inputs, label = data
        y_predict = model(inputs)
        loss = criterion(y_predict, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 预测
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            print(inputs[0].tolist())
            print(type(inputs[0]))
            print(label)
            pdb.set_trace()
            y_pred = model(inputs)
            _, predicted = torch.max(y_pred.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    if correct / total > result:
        torch.save(model.state_dict(), "eeg_model_parameter.pkl")
        result = correct / total

    print(f'Epoch: {epoch + 1}, ACC on test: {correct / total}')
