import torch
import torch.nn as nn
from build_model import get_style_model_and_loss
import torch.optim as optim
from load_img import load_img, show_img


def get_input_param_optimier(input_img):
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(content_img, style_img, input_img, num_epoches):
    print('Building')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(style_img, content_img)

    print('1')
    input_param, optimizer = get_input_param_optimier(input_img)

    print('Optimizing')
    epoch = [0]
    while epoch[0] < num_epoches:
        def closure():
            input_param.data.clamp_(0, 1)
            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            print(epoch[0])
            if epoch[0] % 2 == 0:
                print('run{}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)

    # return input_param.data
    show_img(input_param.data.cpu())