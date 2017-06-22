from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
# %matplotlib inline

url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
			
		last_percent_reported = percent
				
def maybe_download(filename, force=False):
	"""Download a file if not present, and make sure it's the right size."""
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(filename)
	return filename

train_filename = maybe_download('train.tar.gz')
test_filename = maybe_download('test.tar.gz')
extra_filename = maybe_download('extra.tar.gz')

np.random.seed(133)

def maybe_extract(filename, force=False):
	root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
	if os.path.isdir(root) and not force:
		# You may override by setting force=True.
		print('%s already present - Skipping extraction of %s.' % (root, filename))
	else:
		print('Extracting data for %s. This may take a while. Please wait.' % root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	data_folders = root
	print(data_folders)
	return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)

import h5py

# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
	def __init__(self, inf):
		self.inf = h5py.File(inf, 'r')
		self.digitStructName = self.inf['digitStruct']['name']
		self.digitStructBbox = self.inf['digitStruct']['bbox']
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																										 
	# getName returns the 'name' string for for the n(th) digitStruct.
	# 0 for whole image
	# 1, 2, 3,.... for n digits inside given image
	def getName(self,n):
		return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])   

	# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
	def bboxHelper(self,attr):
		if (len(attr) > 1):
			attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
		else:
			attr = [attr.value[0][0]]
		return attr

	# getBbox returns a dict of data for the n(th) bbox. 
	def getBbox(self,n):
		bbox = {}
		bb = self.digitStructBbox[n].item()
		bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
		bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
		bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
		bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
		bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
		return bbox

	def getDigitStructure(self,n):
		s = self.getBbox(n)
		s['name']=self.getName(n)
		return s

	# getAllDigitStructure returns all the digitStruct from the input file.     
	def getAllDigitStructure(self):
		return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

	# Return a restructured version of the dataset (one structure by boxed digit).
	#   Return a list of such dicts :
	#      'filename' : filename of the samples
	#      'boxes' : list of such dicts (one by digit) :
	#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
	#          'left', 'top' : position of bounding box
	#          'width', 'height' : dimension of bounding box
	#
	# Note: We may turn this to a generator, if memory issues arise.
	def getAllDigitStructure_ByDigit(self):
		pictDat = self.getAllDigitStructure()
		result = []
		structCnt = 1
		for i in range(len(pictDat)):
			item = { 'filename' : pictDat[i]["name"] }
			figures = []
			for j in range(len(pictDat[i]['height'])):
				figure = {}
				figure['height'] = pictDat[i]['height'][j]
				figure['label']  = pictDat[i]['label'][j]
				figure['left']   = pictDat[i]['left'][j]
				figure['top']    = pictDat[i]['top'][j]
				figure['width']  = pictDat[i]['width'][j]
				figures.append(figure)
			structCnt = structCnt + 1
			item['boxes'] = figures
			result.append(item)
		return result

train_folders = 'train'
test_folders = 'test'
extra_folders = 'extra'

fin = os.path.join(train_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
train_data = dsf.getAllDigitStructure_ByDigit()

fin = os.path.join(test_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
test_data = dsf.getAllDigitStructure_ByDigit()

fin = os.path.join(extra_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
extra_data = dsf.getAllDigitStructure_ByDigit()

def image_processing(image, boxes):
	image_size = 120
	processed_boxes = []
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
	orig_height =  list(image.shape)[0]
	orig_width =  list(image.shape)[1]
	for bbox in boxes:
		ntop =  (bbox['top'] * image_size) / orig_height
		nleft =  (bbox['left'] * image_size) / orig_width
		nheight = ((bbox['top'] + bbox['height'])* image_size) / orig_height - ntop
		nwidth = ((bbox['left'] + bbox['width'])* image_size) / orig_width - nleft
		processed_boxes.append({'top': ntop, 'left': nleft, 'height': nheight, 'width': nwidth, 'label': bbox['label']})
	kernel = np.ones((3,3),np.uint8)
	processed_image = cv2.erode(image, kernel, iterations = 1)
	return processed_image, processed_boxes

def generate_dataset(data, folder):
	dataset = []
	labels = []
	for i in np.arange(len(data)):
		label = []
		# print("dataset generation step: {}".format(i))
		filename = data[i]['filename']
		fullname = os.path.join(folder, filename)
		im = cv2.imread(fullname)
		cv2.waitKey(10)
		boxes = data[i]['boxes']
		num_digit = len(boxes)
		image, boxes = image_processing(im, boxes)
		for j in range(num_digit):
			test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			test[int(boxes[j]['label'])] = 1
			label.append({'number_of_digits': num_digit, 'label': test,'top': boxes[j]['top'], 'left': boxes[j]['left'], 'height': boxes[j]['height'], 'width':boxes[j]['width']})
		dataset.append(image)
		labels.append(label)
	return np.array(dataset), np.array(labels)

train_dataset, train_labels = generate_dataset(train_data, train_folders)
print(train_dataset.shape, train_labels.shape)

test_dataset, test_labels = generate_dataset(test_data, test_folders)
print(test_dataset.shape, test_labels.shape)

extra_dataset, extra_labels = generate_dataset(extra_data, extra_folders)
print(extra_dataset.shape, extra_labels.shape)

import random

random.seed()

n_labels = 10

valid_dataset = np.concatenate((train_dataset[:400], extra_dataset[:200]), axis=0)
valid_labels = np.concatenate((train_labels[:400], extra_labels[:200]), axis=0)
train_dataset_t = np.concatenate((train_dataset[400:], extra_dataset[200:]), axis=0)
train_labels_t = np.concatenate((train_labels[400:], extra_labels[200:]), axis=0)

print(train_dataset_t.shape, train_labels_t.shape)
print(test_dataset.shape, test_labels.shape)
print(valid_dataset.shape, valid_labels.shape)

pickle_file = '../dataset/SVHN_multi_BBox.npy'

try:
	save = {
		'train_dataset': train_dataset_t,
		'train_labels': train_labels_t,
		'valid_dataset': valid_dataset,
		'valid_labels': valid_labels,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
		}
	np.save(pickle_file, save)
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise
		
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)	