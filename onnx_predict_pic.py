import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from ONNXLearner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import sys
import traceback
# Custom
from log import timer, logger

parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-file", "--file", help="test file name")
parser.add_argument("-o", "--output", help="outputfile name", default="test_out.jpg")
parser.add_argument("-s", "--save", help="whether save",action="store_true")
parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
args = parser.parse_args()

conf = get_config(False)

mtcnn = MTCNN()
print('arcface loaded')

log_load_onnx = timer('Load ONNX Model')

learner = face_learner(conf, True)
learner.threshold = args.threshold
learner.load_state(conf, 'final.onnx', True)

print('learner loaded')

print(args.file)

targets, names = load_facebank(conf)
print('facebank size: '+str(targets.shape))
print('names: '+str(names))
print('facebank loaded')

# inital picture
#frame = cv2.imread('data/raw/dhc/chendh.jpg')
frame = cv2.imread(args.file)

try:
  image = Image.fromarray(frame[...,::-1]) #bgr to rgb
  bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
  bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
  bboxes = bboxes.astype(int)
  bboxes = bboxes + [-1,-1,1,1] # personal choice    
  results, score = learner.infer(conf, faces, targets, args.tta)
    # print(score[0])
  for idx,bbox in enumerate(bboxes):
      if args.score:
          frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
      else:
          if float('{:.2f}'.format(score[idx])) > .98:
              name = names[0]
          else:    
              name = names[results[idx]+1]
          frame = draw_box_name(bbox, names[results[idx] + 1], frame)
  print('output '+ args.output)
  cv2.imwrite(args.output,frame)

except Exception as e:
  error_class = e.__class__.__name__ #取得錯誤類型
  detail = e.args[0] #取得詳細內容
  cl, exc, tb = sys.exc_info() #取得Call Stack
  lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
  fileName = lastCallStack[0] #取得發生的檔案名稱
  lineNum = lastCallStack[1] #取得發生的行號
  funcName = lastCallStack[2] #取得發生的函數名稱
  errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
  print(errMsg)
  pass    

log_load_onnx.end()
