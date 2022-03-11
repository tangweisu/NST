import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision
import PIL.Image as Image
import torchvision.transforms as transforms

img_size = 512


def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).cuda()
    return img

def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()


vgg = torchvision.models.vgg19(pretrained=True).features
vgg = vgg.cuda()

#定义想看哪一层的可视化特征图
content_layers_default = ['conv_4']

def get_feature(content_img, cnn=vgg):


    model = nn.Sequential()
    model = model.cuda()
    i = 1
    for layer in cnn:

        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)
            if name in content_layers_default:
                target = model(content_img)
                print(target.size())
                return target

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

if __name__ == '__main__':
    content_img = load_img('D:\\school.jpg')
    content_img = content_img.cuda()
    target = get_feature(content_img)

    plt.figure()
    for i in range(1, 32):
        plt.subplot(4, 8, i)
        feature = target[:, i, :, :]
        feature = feature.view(feature.shape[1], feature.shape[2])
        plt.imshow(feature.data.cpu())
        plt.xticks([])
        plt.yticks([])
    plt.show()


    # show_img(feature.data.cpu())
