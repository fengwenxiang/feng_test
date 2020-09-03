import torch,cv2
import os,glob
import random,csv
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from    torch.nn import functional as F
from utils import Flatten

try:
    from skimage import data_dir
    from skimage import io
    from skimage import color
    from skimage import img_as_float,transform
    from skimage.transform import resize
except ImportError:
    raise ImportError("This example requires scikit-image")

from FFST import (scalesShearsAndSpectra,
                  inverseShearletTransformSpect,
                  shearletTransformSpect)
from FFST._fft import ifftnc  # centered nD inverse FFT


# NOR_path = 'E:\\NCT-CRC\\train0\\NORMAL'
# TUM_path = 'E:\\NCT-CRC\\train0\\CANCER'
ffst_path="E:\\NCT-CRC\\FFST_img\\"

class Medical(Dataset):

    def __init__(self,norm_path,tum_path,resize,mode):
        super(Medical,self).__init__()
        self.norm_path= norm_path
        self.tum_path= tum_path
        self.resize = resize
        self.images,self.labels=self.load_csv("images.csv")


        if mode=="train":  #60%
            self.images=self.images[:int(0.6*len(self.images))]
            self.labels=self.labels[:int(0.6*len(self.labels))]
        elif mode=="val":  #20%  =60%->80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  #20%  =80%->100%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self, filename):
        if os.path.exists( filename) == 0:
            images=[]
            for path in (self.norm_path,self.tum_path):
                for img_name in  os.listdir(path):
                    images+=glob.glob(path+"\\"+img_name)
            random.shuffle(images)

            with open(filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:
                    if "NORM" in img:
                        label=1
                    if  "TUM" in img:
                        label = 0
                    writer.writerow([img,label])
                print("writen into csv file:", filename)

        images,labels=[],[]
        with open( filename)as f:
            reader = csv.reader(f)
            for row in reader:
                img,label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        # print("images",images)
        # print(len(images))
        assert len(images) == len(labels)

        return images,labels

    def __len__(self):
        return len(self.images)

    def load_img_data(self,filename):
        # print("filename=",filename)
        name = filename.split(os.sep)[-1]
        strname=(ffst_path+name).replace('.tif', '.npy')
        if os.path.exists(strname):
            b = np.load(strname)     #b[224,224,58]
            mm=b[...,0:29]
            phh =b[...,29:58]
            # print("yes")
        else:
            # print("nono", strname)
            rgb = io.imread(filename)
            gray = color.rgb2gray(rgb)
            image = img_as_float(gray)
            imag = resize(image, (224, 224))
            ST, _ = shearletTransformSpect(imag, realCoefficients=False)    #ST[224,224,29]
            mm = np.abs(ST)
            phh = np.angle(ST)
            cat=np.concatenate((mm,phh),axis=2)
            np.save(strname, cat)
        return mm[...,0:3], phh[...,0:3]


    def np_normlize(self,scr):
        """
        归一化 ，norm_type=cv2.NORM_MINMAX ，并对图片进行维度变换[imgsize,imgsize,3]=>[3,imgsize,imgsize]
        :param scr:numpy 类型,[imgsize,imgsize,3]
        :return: numpy 类型,[3,imgsize,imgsize]
        """
        # img0 = cv2.cvtColor(scr, cv2.COLOR_BGR2RGB)  # convert to RGB
        img0 = cv2.resize(scr, dsize=(self.resize, self.resize), interpolation=cv2.INTER_CUBIC)
        img0 = np.array(img0, dtype=np.float32)
        result0 = np.zeros(img0.shape, dtype=np.float32)
        cv2.normalize(img0, result0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        result0 = torch.tensor(result0).float().permute(2, 0, 1)
        return result0


    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]
        rgb = io.imread(img)                    #将图片数据读进rgb，numpy类型，[imgsize,imgsize,3]

        mm,phh = self.load_img_data(img)                #ffst变换加载，幅度和相位，mm numpy 类型 [imgsize,imgsize,3（可调整）],phh  numpy 类型[imgsize,imgsize,3（可调整]

        result0=self.np_normlize(rgb)
        result1=self.np_normlize(mm)
        result2=self.np_normlize(phh)
        #composite 9个维度 ；0 1 2 是图片/ 3 4 5是幅度 idx=0，1，2/ 6 7 8 是相位 idx =0，1，2
        composite=torch.cat((result0,result1,result2),0)  # (result0,result1,result2) 都numpy 类型[3,imgsize,imgsize]=>[9,imgsize,imgsize]
        label = torch.tensor(label)
        return img,composite, label

class ConBlk(nn.Module):

    def __init__(self):
        super(ConBlk,self).__init__()
        self.conv1 = nn.Conv2d(3, 36, kernel_size=3, stride=1, padding=1)
        self.pool1=nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(36)
        self.conv2 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fla=Flatten()

    def forward(self,x):
        x1=F.relu(self.bn1(self.pool1(self.conv1(x))))
        x2=F.relu(self.bn2(self.pool2(self.conv2(x1))))
        x3=F.relu(self.pool3(self.conv3(x2)))
        out=self.fla(x3)
        return out

class CovNet(nn.Module):
    def __init__(self):
        super(CovNet, self).__init__()
        self.blk1=ConBlk()
        self.blk2=ConBlk()
        self.blk3=ConBlk()

        self.outlayer = nn.Sequential(
            nn.Linear(6912, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()

        )

    def forward(self, x):

        x1=self.blk1(x[:,0:3, ...])
        x2 = self.blk2(x[:,3:6, ...])
        x3 = self.blk3(x[:,6:9, ...])
        out=torch.cat((x1,x2,x3),1)
        # print("out.shape:::",out.shape)
        out=self.outlayer(out)
        return out

def test():

    # blk = ConBlk()
    tmp = torch.randn(16,9, 64, 64)
    # out = blk(tmp)
    # print('block:', out.shape)
    # print(blk)

    model=CovNet()
    out2=model(tmp)
    print('=CovNet:', out2.shape)

def main():
    from visdom import Visdom
    import time
    import torchvision

    # viz = Visdom()

    db =Medical(NOR_path,TUM_path, 224, "train")
    x,y=next(iter(db))
    # xx = x[4:7, ...]
    # viz.image(xx, win="sample_x", opts=dict(title="sample_x"))
    print("sample",x.shape,y.shape,y)

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=4)
    print("loader",loader.__len__())
    i=0
    for x, y in loader:
        print("第",i,"个")
        i=i+1

        xx=x[:,0:3, ...]
        # print("x.shape=",xx.shape)
        # viz.images( xx, nrow=4, win="batch", opts=dict(title="batch"))
        # viz.text(str(y.numpy()), win="lablel", opts=dict(title="batch-y"))
        # time.sleep(20)


    """
    xx=x[0:3,...].permute(1,2,0)
    plt.figure()
    plt.imshow(xx, interpolation='nearest', cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()
    """



    # print("xx.shape",xx.shape)

if __name__ == '__main__':
    main()