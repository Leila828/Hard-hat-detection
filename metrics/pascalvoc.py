###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys
import csv

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                '%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isfile(arg) is False and os.path.isfile(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isfile(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(bbox_file,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    with open(bbox_file, newline='') as csvfile:
        bbox_reader = csv.reader(csvfile, delimiter=',')
        next(bbox_reader, None)
        for row in bbox_reader:
            nameOfImage = row[0]
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (row[3])  # class
                x = float(row[4])
                y = float(row[5])
                w = float(row[6])
                h = float(row[7])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (row[1])  # class
                confidence = float(row[2])
                x = float(row[3])
                y = float(row[4])
                w = float(row[5])
                h = float(row[6])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1 (beta)'

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description='This project applies the most popular metrics used to evaluate object detection '
    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# formatter_class=RawTextHelpFormatter)
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument(
    '-gt',
    '--gtfile',
    dest='gtFile',
    default=os.path.join(currentPath, 'groundtruths'),
    metavar='',
    help='folder containing your ground truth bounding boxes')
parser.add_argument(
    '-det',
    '--detfile',
    dest='detFile',
    default=os.path.join(currentPath, 'detections'),
    metavar='',
    help='folder containing your detected bounding boxes')
# Optional
parser.add_argument(
    '-t',
    '--threshold',
    dest='iouThreshold',
    type=float,
    default=0.5,
    metavar='',
    help='IOU threshold. Default 0.5')
parser.add_argument(
    '-gtformat',
    dest='gtFormat',
    metavar='',
    default='xywh',
    help='format of the coordinates of the ground truth bounding boxes: '
    '(\'xywh\': <left> <top> <width> <height>)'
    ' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
    '-detformat',
    dest='detFormat',
    metavar='',
    default='xywh',
    help='format of the coordinates of the detected bounding boxes '
    '(\'xywh\': <left> <top> <width> <height>) '
    'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
    '-gtcoords',
    dest='gtCoordinates',
    default='abs',
    metavar='',
    help='reference of the ground truth bounding box coordinates: absolute '
    'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-detcoords',
    default='abs',
    dest='detCoordinates',
    metavar='',
    help='reference of the ground truth bounding box coordinates: '
    'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
    '-imgsize',
    dest='imgSize',
    metavar='',
    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument(
    '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
parser.add_argument(
    '-np',
    '--noplot',
    dest='showPlot',
    action='store_false',
    help='no plot is shown during execution')
args = parser.parse_args()

iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFile, '-gt/--gtfile', errors):
    gtFile = ValidatePaths(args.gtFile, '-gt/--gtfile', errors)
else:
    # errors.pop()
    gtFile = os.path.join(currentPath, 'groundtruths.csv')
    if os.path.isfile(gtFile) is False:
        errors.append('file %s not found' % gtFile)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
# Detection folder
if ValidateMandatoryArgs(args.detFile, '-det/--detfile', errors):
    detFile = ValidatePaths(args.detFile, '-det/--detfile', errors)
else:
    # errors.pop()
    detFile = os.path.join(currentPath, 'detections.csv')
    if os.path.isfile(detFile) is False:
        errors.append('file %s not found' % detFile)
if args.savePath is not None:
    savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
else:
    savePath = os.path.join(currentPath, 'results')
# Validate savePath
# If error, show error messages
if len(errors) != 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Check if path to save results already exists and is not empty
if os.path.isfile(savePath) and os.listdir(savePath) :
    key_pressed = ''
    while key_pressed.upper() not in ['Y', 'N']:
        print(f'Folder {savePath} already exists and may contain important results.\n')
        print(f'Enter \'Y\' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!')
        print(f'Or enter \'N\' to abort and choose another folder to save the results.')
        key_pressed = input('')

    if key_pressed.upper() == 'N':
        print('Process canceled')
        sys.exit()

# Clear folder and save results
shutil.rmtree(savePath, ignore_errors=True)
os.makedirs(savePath)
# Show plot during execution
showPlot = args.showPlot

# print('iouThreshold= %f' % iouThreshold)
# print('savePath = %s' % savePath)
# print('gtFormat = %s' % gtFormat)
# print('detFormat = %s' % detFormat)
# print('gtFile = %s' % gtFile)
# print('detFile = %s' % detFile)
# print('gtCoordType = %s' % gtCoordType)
# print('detCoordType = %s' % detCoordType)
# print('showPlot %s' % showPlot)

# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    gtFile, True, gtFormat, gtCoordType, imgSize=imgSize)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    detFile, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
allClasses.sort()

evaluator = Evaluator()
acc_AP = 0
validClasses = 0

# Plot Precision x Recall curve
detections = evaluator.PlotPrecisionRecallCurve(
    allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=iouThreshold,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation,
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
    savePath=savePath,
    showGraphic=showPlot)

f = open(os.path.join(savePath, 'results.txt'), 'w')
f.write('Object Detection Metrics\n')
f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
f.write('Average Precision (AP), Precision and Recall per class:')

# each detection is a class
for metricsPerClass in detections:

    # Get metric values per each class
    cl = metricsPerClass['class']
    ap = metricsPerClass['AP']
    precision = metricsPerClass['precision']
    recall = metricsPerClass['recall']
    totalPositives = metricsPerClass['total positives']
    total_TP = metricsPerClass['total TP']
    total_FP = metricsPerClass['total FP']

    if totalPositives > 0:
        validClasses = validClasses + 1
        acc_AP = acc_AP + ap
        prec = ['%.2f' % p for p in precision]
        rec = ['%.2f' % r for r in recall]
        ap_str = "{0:.2f}%".format(ap * 100)
        # ap_str = "{0:.4f}%".format(ap * 100)
        print('AP: %s (%s)' % (ap_str, cl))
        f.write('\n\nClass: %s' % cl)
        f.write('\nAP: %s' % ap_str)
        f.write('\nPrecision: %s' % prec)
        f.write('\nRecall: %s' % rec)

mAP = acc_AP / validClasses
mAP_str = "{0:.2f}%".format(mAP * 100)
print('mAP: %s' % mAP_str)
f.write('\n\n\nmAP: %s' % mAP_str)
