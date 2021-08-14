#pip install torchinfo

from torchinfo import summary
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
#from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-o", "--output", help="outputfile name", default="test_out.jpg")
parser.add_argument("-s", "--save", help="whether save",action="store_true")
parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
args = parser.parse_args()

conf = get_config(False)

#mtcnn = MTCNN()
print('arcface loaded')

learner = face_learner(conf, True)
learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')


model = learner.model
batch_size = 1
summary(model, input_size=(batch_size, 3, 112, 112))
