import torch.nn as nn
import loss
import torchvision

vgg = torchvision.models.vgg19(pretrained=True).features
vgg = vgg.cuda()
# print(vgg)

content_layers_default = ['conv_1']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_loss(style_img, content_img, cnn=vgg,
                             style_weight=5000,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default
                             ):
    print('ready get loss')
    print()
    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential()
    model = model.cuda()
    gram = loss.Gram()
    gram = gram.cuda()

    print('ready go cuda')
    i = 1
    for layer in cnn:
        #判断layer是否是后面的类型
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)
            print('finish add conV')

            #定义提取内容的layer
            if name in content_layers:
                target = model(content_img)
                content_loss = loss.Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)
                print('finish add conLayer')

            #定义提取风格的layer
            if name in style_layers:
                target = model(style_img)
                target = gram(target)
                style_loss = loss.Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)
                print('finish add styLayer')


            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    print(model)
    return model, style_loss_list, content_loss_list

