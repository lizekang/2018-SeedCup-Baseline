import torch
from word_idx import word_idx
from class_idx import class_idx
from DataLoader import MyTestSet
from config import Config
from model import FC
import torch.backends.cudnn as cudnn
import torch.nn as nn

opt = Config()
word_with_idx = word_idx(opt)
class_with_idx = class_idx(opt)

print('Loading data...')
testset = MyTestSet(word_with_idx, class_with_idx)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=opt.TEST_BATCH_SIZE)

print('Loading model')
net = FC(opt)
net = torch.load(opt.MODEL_FILE)['net']
if opt.USE_CUDA:
    net.cuda()
    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
print('Loading model sucessfully')


def test():
    with open(opt.ANS_FILE, 'w')as f:
        f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
        for batch_idx, (inputs, ids)in enumerate(testloader):
            if opt.USE_CUDA:
                inputs = inputs.cuda()
            output_1, output_2, output_3 = net(inputs)
            class_1 = torch.max(output_1, dim=1)[1]
            class_1 = class_1.cpu().numpy()
            class_2 = torch.max(output_2, dim=1)[1]
            class_2 = class_2.cpu().numpy()
            class_3 = torch.max(output_3, dim=1)[1]
            class_3 = class_3.cpu().numpy()
            for i in range(len(ids)):
                classes = [class_1[i], class_2[i], class_3[i]]
                # print(classes)
                classes = class_with_idx.idx_to_class(classes)
                # print(classes)
                answer = ids[i]+'\t'+classes[0]+'\t' + \
                    classes[1]+'\t'+classes[2]+'\n'
                f.write(answer)


test()
