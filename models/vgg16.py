import torch.nn as nn
import torch

__all__ = ['Vgg16', 'vgg16']

class Vgg16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(Vgg16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 0
            nn.ReLU(inplace=True), # 1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 2
            nn.ReLU(inplace=True), # 30
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), # 4
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 5
            nn.ReLU(inplace=True), # 6
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 7
            nn.ReLU(inplace=True), # 8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), # 9
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 10
            nn.ReLU(inplace=True), # 11
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 12
            nn.ReLU(inplace=True), # 13
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 14
            nn.ReLU(inplace=True), # 15
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), # 16
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 17
            nn.ReLU(inplace=True), # 18
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 19
            nn.ReLU(inplace=True), # 20
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 21
            nn.ReLU(inplace=True), # 22
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), # 23
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 24
            nn.ReLU(inplace=True), # 25
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 26
            nn.ReLU(inplace=True), # 27
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # 28
            nn.ReLU(inplace=True), # 29
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), # 30
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 18
        # x = torch.flatten(x, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 19
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096), # 20
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes), # 21
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, server=True, partition=0):
        if server == True:
            if partition == 0:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 1:
                x = self.features[2:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 2:
                x = self.features[4:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 3:
                x = self.features[5:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 4:
                x = self.features[7:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 5:
                x = self.features[9:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 6:
                x = self.features[10:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 7:
                x = self.features[12:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 8:
                x = self.features[14:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 9:
                x = self.features[16:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 10:
                x = self.features[17:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 11:
                x = self.features[19:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 12:
                x = self.features[21:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 13:
                x = self.features[23:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 14:
                x = self.features[24:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 15:
                x = self.features[26:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 16:
                x = self.features[28:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 17:
                x = self.features[30:](x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 18:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            elif partition == 19:
                x = self.classifier(x)
            elif partition == 20:
                x = self.classifier[3:](x)
            elif partition == 21:
                x = self.classifier[6:](x)
            elif partition == 22:
                x = x
            else:
                print('Please give the right partition point.')
        else:
            if partition == 0:
                x = x
            elif partition == 1:
                x = self.features[0:2](x)
            elif partition == 2:
                x = self.features[0:4](x)
            elif partition == 3:
                x = self.features[0:5](x)
            elif partition == 4:
                x = self.features[0:7](x)
            elif partition == 5:
                x = self.features[0:9](x)
            elif partition == 6:
                x = self.features[0:10](x)
            elif partition == 7:
                x = self.features[0:12](x)
            elif partition == 8:
                x = self.features[0:14](x)
            elif partition == 9:
                x = self.features[0:16](x)
            elif partition == 10:
                x = self.features[0:17](x)
            elif partition == 11:
                x = self.features[0:19](x)
            elif partition == 12:
                x = self.features[0:21](x)
            elif partition == 13:
                x = self.features[0:23](x)
            elif partition == 14:
                x = self.features[0:24](x)
            elif partition == 15:
                x = self.features[0:26](x)
            elif partition == 16:
                x = self.features[0:28](x)
            elif partition == 17:
                x = self.features[0:30](x)
            elif partition == 18:
                x = self.features(x)
            elif partition == 19:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
            elif partition == 20:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier[0:3](x)
            elif partition == 21:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier[0:6](x)
            else:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16(num_classes=1000, pretrained=True, progress=True):
    file = 'https://download.pytorch.org/models/vgg16-397923af.pth'
    model = Vgg16(num_classes)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(file, progress=progress)
        model.load_state_dict(state_dict)

    return model

if __name__ == '__main__':
    print('test partition points in vgg16!!!')

    import json
    import torchvision.transforms as transforms
    from PIL import Image

    with open("imagenet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}

    model = vgg16()
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    min_img_size = 224
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    img = Image.open('Golden_Retriever_Hund_Dog.jpg')
    img = transform_pipeline(img)
    img = img.unsqueeze(0)

    for partition in range(23):
        with torch.no_grad():
            intermediate = model(img.cuda(), server=False, partition=partition)
            prediction = model(intermediate, server=True, partition=partition)

            prediction = torch.argmax(prediction)

            print('partition point ', partition, labels[prediction.item()])
