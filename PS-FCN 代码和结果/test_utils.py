import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils, time_utils 

def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    return disp_intv, save_intv

def test(args, split, loader, model, log, epoch, recorder):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))

    with torch.no_grad():
        #loader就是DataLoade处理的数据，i是第几个obj，sample是第i个obj对应的item
        for i, sample in enumerate(loader):
            #这一步img的形状没变，就是改成tensor了
            data = model_utils.parseData(args, sample, timer, split)
            #这个函数应该是得到输入的吧，直接传给模型了
            input = model_utils.getInput(args, data)

            #得到模型输出数据
            out_var = model(input); timer.updateTime('Forward')
            #计算准确率，应该是一个loss之类的
            acc = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data) 
            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            #在第几轮显示训练数据
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                #打印训练数据
                log.printItersSummary(opt)

            #在第几轮保存模型
            if iters % save_intv == 0:
                pred = (out_var.data + 1) / 2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                log.saveNormalResults(masked_pred, split, epoch, iters)

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

