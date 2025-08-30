
class Config:
    def __init__(self,task):
        if task == "synapse":
            self.base_dir = '/home/chenzh/FILE1/Datasets/Synapse'
            self.save_dir = '/home/chenzh/FILE1/DHC/DBANet/synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
        elif task == "amos":
            self.base_dir = '/home/chenzh/FILE1/Datasets/amos22'
            self.save_dir = '/home/chenzh/FILE1/DHC/DBANet/amos_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
        else: # word
            self.base_dir = '/home/chenzh/FILE1/Datasets/wordv01'
            self.save_dir = '/home/chenzh/FILE1/DHC/DBANet/word_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 17
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
