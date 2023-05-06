import os
import torch
import torch.nn as nn

#鸟用没有
def getInput(args, data):
    input_list = [data['input']]
    #好家伙光照方向直接append进输入里了
    if args.in_light: input_list.append(data['l'])
    return input_list
#真正获取输入数据
def parseData(args, sample, timer=None, split='train'):

    #得到预处理后的大图像、法向量图、mask图
    input, target, mask = sample['img'], sample['N'], sample['mask']

    #以下两句屁用没有
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); 


    #包装成具有数据和梯度的类，现在应该自动带有梯度了
    input_var  = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False);

    if timer: timer.updateTime('ToGPU')
    data = {'input': input_var, 'tar': target_var, 'm': mask_var}

    #增加光照条件
    if args.in_light:
        #把光照扩展成和img形状一样的
        light = sample['light'].expand_as(input)
        if args.cuda: light = light.cuda()
        light_var = torch.autograd.Variable(light);
        data['l'] = light_var
    return data 

def getInputChanel(args):
    print('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        print('[Network Input] Adding Light direction as input')
        c_in += 3
    print('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records, 
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
