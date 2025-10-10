#alexnet original
import torch
import torch.nn

#alexnet original
import torch
import torch.nn as nn

#nn.Module -> 모든 신경망의 부모클래스, nn.module초기화 해야지 신경망모델이라고 인식

#PyTorch 구현 (1-GPU 통합, 논문 원형 충실: LRN + Overlapping Pooling)
class AlexNetOriginal(nn.Module):
    def __init__(self, num_classes = 1000):

        #super(AlexNet, self).__init__()
        super().__init__() #(nn.Module)의 초기화 함수를 실행해서 모델 안의 Layer들을 추적하고 관리할 준비
        # LRN: n=5, k=2, alpha=1e-4, beta=0.75 (논문 설정)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

        #Normalization -> after Conv1, Conv2
        #Relu -> everywhere
        #MaxPooling -> after Conv1, Conv2, Conv 5
        #Dropout -> first two linear layers with p=0.5


        
        self.features = nn.Sequential(
            #224 * 224 -> 55 * 55
            nn.Conv2d(3,96, kernel_size = 11, stride = 4, padding = 2), # cv1
            nn.ReLU(inplace=True),#입력값 덮어쓰기
            self.lrn1,
            #55 * 55 -> 27 * 27
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            #27 * 27 -> 27 * 27
            nn.Conv2d(96, 256, kernel_size = 5, padding = 2), #cv2
            nn.ReLU(inplace=True),
            self.lrn2,
            #27 * 27 -> 13 * 13 
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            #13 * 13 -> 13 * 13 
            nn.Conv2d(256, 384, kernel_size=3, padding = 1), #cv 3
            nn.ReLU(inplace=True),

            #13 * 13 -> 13 * 13 
            nn.Conv2d(384, 384, kernel_size= 3, padding = 1),#cv4
            nn.ReLU(inplace=True),



            nn.Conv2d(384, 256, kernel_size = 3,padding = 1),# cv 5
            nn.ReLU(inplace=True),
            #13 * 13 -> 6 * 6
            nn.MaxPool2d(kernel_size = 3, stride = 2)


        )

        self.classifier = nn.Sequential(
            # output_channel * H * W
            nn.Dropout(p=0.5),
            #Conv 출력을 1차원으로 (batch_size, channel * h * w)
            #torch.flatten(x, 1) 
            nn.Linear(256 * 6 * 6, 4096), # linear 1
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096 , 4096), # linear 2
            nn.ReLU(inplace=True),


            nn.Linear(4096 , num_classes), # linear 3

        )
    def forward(self, x):
      x = self.features(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x


#AlexNet with batchnorm version

class AlexNet_BN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


