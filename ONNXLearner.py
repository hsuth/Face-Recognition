from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import onnxruntime 

def to_numpy(tensor):
      return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class ONNXModel(object):
    def __init__(self, conf):
        self.filename=""
        self.conf=conf
        
    def load_state_dict(self, filename):
        self.filename=str(filename)
        print(self.filename)
        self.ort_session = onnxruntime.InferenceSession(self.filename)       
        print(self.filename)
    
     
    #img_y is pytorch tensor type
    def __call__(self, img_y):
        self.ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(img_y)}
        self.ort_outs = self.ort_session.run(None, self.ort_inputs)
        result = torch.from_numpy(self.ort_outs[0]).to(self.conf.device)
        return result
      
    
class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.model = ONNXModel(conf)            
        self.threshold = conf.threshold
    
    def load_state(self, conf, fixed_str, from_save_folder=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        self.model.load_state_dict(save_path/'model_{}'.format(fixed_str))
          
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               
