import warnings
warnings.filterwarnings("ignore")
import torch,cv2
from torch import optim,nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

from Medical import Medical,ConBlk,CovNet
from visdom import Visdom
import time
import torchvision

try:
    from skimage import data_dir
    from skimage import io
    from skimage import color
    from skimage import img_as_float,transform
    from skimage.transform import resize
except ImportError:
    raise ImportError("This example requires scikit-image")

NOR_path = 'E:\\NCT-CRC\\train\\NORMAL'
TUM_path = 'E:\\NCT-CRC\\train\\CANCER'

batchsz=32
lr=1e-3
epochs=30
imgsize=64

device=torch.device("cuda")
torch.manual_seed(1234)
train_db=Medical(NOR_path,TUM_path,imgsize,mode="train")
val_db=Medical(NOR_path,TUM_path,imgsize,mode="val")
test_db=Medical(NOR_path,TUM_path,imgsize,mode="test")
train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,
                        num_workers=4)
val_loader=DataLoader(val_db,batch_size=batchsz, num_workers=2)
test_loader=DataLoader(test_db,batch_size=batchsz, num_workers=2)

viz=visdom.Visdom()

def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    print("total:",total)
    for r,x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            # print("logitslogits logits  :",logits)
            pred = torch.squeeze(logits.ge(0.5).long(),1)  # 以0.5为阈值进行分类
        correct +=torch.eq(pred, y).sum().float().item()
        # print(" correct  :", correct)

    return correct/ total

def main():
    model = CovNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
    criteon = nn.BCELoss().to(device)
    viz.line([0],[-1],win="loss",opts=dict(title="loss"))
    viz.line([0], [-1], win="val_acc", opts=dict(title="val_acc"))
    best_acc, best_epoch = 0, 0
    global_step = 0
    # print(model)
    for epoch in range(epochs):

        for step, (r,x, y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device).float()
            logits = model(x)
            # print("y.type()", y.type(), y)
            # print("logits.type()", logits.type(),logits)
            loss = criteon(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%10==0:
                print("epoch:", epoch, "step:", step, "loss:", loss.item())
            viz.line([loss.item()], [global_step], win="loss", update="append")
            global_step=global_step+1
        if epoch % 1 == 0:
            # correct = 0
            # total = len(val_loader.dataset)
            # for r,x, y in val_loader:
            #
            #     x, y = x.to(device), y.to(device)
            #     # print("y.type()", y.type(), y)
            #     with torch.no_grad():
            #         logits = model(x)
            #         # print("logits.type()", logits.type(),logits)
            #         pred = logits.argmax(dim=1)
            #         # print("pred.type()", pred .type(),pred)
            #     correct += torch.eq(pred, y).sum().float().item()
            # val_acc=correct / total
            val_acc = evalute(model, val_loader)
            viz.line([val_acc], [global_step], win="val_acc", update="append")
            print("epoch:", epoch, "val_acc:", val_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), "medical_best.mdl")

    print("best acc:", best_acc, "best epoch:", best_epoch)
    model.load_state_dict(torch.load("medical_best.mdl"))
    print("loaded from ckpt!")

    test_acc = evalute(model, test_loader)
    # correct = 0
    # global_step=0
    # total = len(test_loader.dataset)
    # for r, x, y in test_loader:
    #     x, y = x.to(device), y.to(device)
    #     with torch.no_grad():
    #         global_step=global_step+1
    #         logits = model(x)
    #         # print("logits:",logits,logits.type())
    #         # print("y:",y,y.type())
    #         loss = criteon(logits, y)/len(y)
    #         # viz.line([loss.item()], [global_step], win="loss", update="append")
    #         # print("our loss:",loss)
    #         pred = logits.argmax(dim=1)
    #         # print("pred=",pred)
    #     correct += torch.eq(pred, y).sum().float().item()
    # test_acc=correct / total
    print("test acc:", test_acc)


            # x, y = x.to(device), y.to(device).float()


            # print("r",r)
            # print("yyyyy=",y)
            # print(x.shape)
            # for i in range(batchsz):
                # pt=r[i]
                # print(pt)
                # rgb = io.imread(pt)
                # print("5555",rgb.max(),rgb.min(),rgb.mean())
                # cv2.imshow('image', rgb)
                #
                # img0 = cv2.resize(rgb, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
                # print("5556", img0.max(), img0.min(), img0.mean())
                #
                # img0 = np.array(img0, dtype=np.float32)
                # print("5557", img0.max(), img0.min(), img0.mean())
                # result0 = np.zeros(img0.shape, dtype=np.float32)
                #
                #
                #
                # cv2.normalize(img0, result0, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)    # result0[224,224,3]
                # print("5558", result0.max(),result0.min(), result0.mean(),type(result0))

                # lz= np.zeros(img0.shape, dtype=np.float32)


                # min=img0.min()
                # max=img0.max()
                # lz = (img0 -  min) / ( max - min)
                # if (lz==result0).all():
                #     print("lz==result0")
                # else:
                #     print("lz not")
                #     print("55582", result0.max(), result0.min(), result0.mean())
                #     print(result0[0:20,0,0])
                #     print("55592", lz.max(), lz.min(), lz.mean(),lz.dtype)
                #     print(lz[0:20, 0, 0])
                #     hello=lz*(max-min)+min
                #     if (hello == img0).all():
                #         print("yesysysy")
                #     else:
                #         print("mmmm")






                # result0 = torch.tensor(result0).float().permute(2, 0, 1)
                # print("5559", result0.max(), result0.min(), result0.mean())
                #








                # tmp=x[i]
                # tmp=torch.squeeze(tmp,0)
                # print("tmp.shape1", tmp.shape)
                #
                # tmp=tmp[7, ...]
                # print("tmp.shape",tmp.shape)
                # io.imsave("d:\\78.png",tmp)

                # if (result0==tmp).all():
                #     print("yes,result0==tmp")
                # else:
                #     print("no")


                # print("tmp:",i,"的",tmp.max(),tmp.min(),tmp.mean(),tmp.type())
                # print("here",tmp.shape)
                #
                # plt.figure()
                # plt.imshow(tmp, interpolation='nearest', cmap=plt.cm.gray)
                # plt.colorbar()
                # plt.show()



            # viz.line([loss.item()], [global_step], win="loss", update="append")
            # global_step += 1

        # if epoch % 1 == 0:









        # if epoch % 1 == 0:
        #     val_acc = evalute(model, val_loader)
        #     viz.line([val_acc], [global_step], win="val_acc", update="append")
        #     print("epoch:", epoch, "val_acc:", val_acc)
        #     if val_acc > best_acc:
        #         best_epoch = epoch
        #         best_acc = val_acc
        #         torch.save(model.state_dict(), "best.mdl")
        #


    # print("best acc:", best_acc, "best epoch:", best_epoch)
    # model.load_state_dict(torch.load("best.mdl"))
    # print("loaded from ckpt!")

    # test_acc = evalute(model, test_loader)
    # print("test acc:", test_acc)



if __name__ == '__main__':
    main()