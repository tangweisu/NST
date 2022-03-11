from torch.autograd import Variable
from run_code import run_style_transfer
from load_img import load_img, show_img

style_img = load_img('D:\\udnie.jpg')
show_img = Variable(style_img).cuda()
content_img = load_img('D:\\school.jpg')
content_img = Variable(content_img).cuda()
input_img = content_img.clone()
run_style_transfer(content_img, style_img, input_img, num_epoches=300)
# print(out.type())
# print(out.size())
# print(out)
# show_img(out)