class myArg():
    def __init__(self):
        self.cuda=True
        self.time_sync=False
        self.workers=0
        self.seed=0
        self.fuse_type='max'
        self.normalize=False
        self.in_light=True
        self.use_BN=False
        self.train_img_num=32
        self.in_img_num=32
        self.start_epoch=1
        self.epoches=30
        self.resume=None
        self.retrain="data/models/PS-FCN_B_S_32.pth.tar"
        self.save_root='data/Training/'
        self.item='calib'
        self.run_model=True
        self.benchmark='DiLiGenT_main'
        self.bm_dir='data/datasets/DiLiGenT/pmsData'
        self.model='PS_FCN_run'
        self.test_batch=1
        self.test_intv=1
        self.test_disp=1
        self.test_save=1