import torch
class Config(object):
    def __init__(self):
        self.USE_CUDA           =       torch.cuda.is_available()
        self.NUM_EPOCHS         =       100
        self.TRAIN_BATCH_SIZE   =       128
        self.VAL_BATCH_SIZE     =       128
        self.TEST_BATCH_SIZE    =       128
        self.MODEL_FILE         =       './model.t7'
        
        self.TRAIN_FILE         =       './data/train_a.txt'
        self.VAL_FILE           =       './data/valid_a.txt'
        self.TEST_FILE          =       './data/test_a.txt'
        self.ANS_FILE           =       './data/ans.txt'
        self.LR                 =       1e-3

        self.NUM_CLASS_1        =       10
        self.NUM_CLASS_2        =       64
        self.NUM_CLASS_3        =       125
        self.EMBEDDING_DIM      =       100
        self.VOCAB_SIZE         =       77810

        self.TITLE_DIM          =       512
        self.LINER_HID_SIZE     =       1024
        self.SENT_LEN           =       100

    def get_lr(self,epoch):
        if (epoch+1)%10==0 and self.LR>1e-7:
            self.LR*=0.1
        print("learning rate:",self.LR)
        return self.LR
