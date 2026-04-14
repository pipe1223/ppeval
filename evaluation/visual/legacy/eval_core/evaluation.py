from tianhuieval.captioning import evaluate_captioning
from tianhuieval.classification import evaluate_classification
from tianhuieval.segmentation import evaluate_segmentation
from tianhuieval.detection import evaluate_detection
from tianhuieval.detection_obb import evaluate_detection2
from tianhuieval.disaster import eva_disaster
from tianhuieval.vqa import evaluate_vqa
import tianhuieval.const as const
import numpy as np
import json
import argparse
import os
import re
import shutil
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

temp_folder = './tmp/'

cls_tags = const.CLS_TAGS
cap_tags = const.CAP_TAGS
vqa_tags = const.VQA_TAGS
det_tags = const.DET_TAGS
seg_tags = const.SEG_TAGS

disaster_tags = ['Fire', 'Flood', 'Volcano', 'landslide']


def extract_roi (input_string, pattern = r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\|<(\d+)>"):
    # Regular expression pattern to capture the required groups
    pattern = pattern
    # Find all matches
    matches = re.findall(pattern, input_string)
    
    # Extract the values
    extracted_values = [match for match in matches]

    return extracted_values

def round_up_to_nearest(x):
    if x <= 0:
        return 0  # Handle non-positive numbers
    magnitude = 10 ** (len(str(x)) - 1)
    if x == magnitude:
        return x
    else:
        return magnitude * 10


def calculate_iou_hbb(box1, box2):
    """计算两个HBB之间的IoU"""


    x_min_inter = float(max(float(box1[0]), float(box2[0])))
    y_min_inter = float(max(float(box1[1]), float(box2[1])))
    x_max_inter = float(min(float(box1[2]), float(box2[2])))
    y_max_inter = float(min(float(box1[3]), float(box2[3])))

    inter_area = float(max(0, x_max_inter - x_min_inter)) * float(max(0, y_max_inter - y_min_inter))
    box1_area = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
    box2_area = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
    union_area = box1_area + box2_area - inter_area+0.1

    iou = float(inter_area) / float(union_area)
    return iou

def prepare_detection_format(file_name, analytic = False):

    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    print ("save HBB", folder_name)
    temp_dir = temp_folder+folder_name+'/det/'
    file_json = open(file_name)
    json_obj =  json.load(file_json)
    
    try:
        info_json = json_obj[const.JSON_INFO]
        result_task = info_json[const.JSON_TASK]
    except:
        result_task = json_obj[const.JSON_TASK]
    data_json = json_obj[const.JSON_DATA]

    #delete old data
    if os.path.exists(temp_dir+const.GT):
        shutil.rmtree(temp_dir+const.GT)
    if os.path.exists(temp_dir+const.DET_RES):
        shutil.rmtree(temp_dir+const.DET_RES)

    #create new data for new test
    if not os.path.exists(temp_dir+const.GT):
        os.makedirs(temp_dir+const.GT)
    if not os.path.exists(temp_dir+const.DET_RES):
        os.makedirs(temp_dir+const.DET_RES)


    max_answer = 0
    max_gt = 0
    
    multiplyer_ans = 1
    multiplyer_gt = 1
    
    for data_j in data_json:
        answer_roi = data_j[const.JSON_ANS]
        gt_roi = data_j[const.JSON_GT]
        if not isinstance(gt_roi, list):  
            gt_roi = extract_roi(gt_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')
    
        if not isinstance(answer_roi, list):
            answer_roi = extract_roi(answer_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')
    
        for gt in gt_roi:
            if max_gt < int(max(gt)):
                max_gt = int(max(gt))
    
        for ans in answer_roi:
            if max_answer < int(max(ans)):
                max_answer = int(max(ans))
    
    
    max_answer = round_up_to_nearest(max_answer)
    max_gt = round_up_to_nearest(max_gt)
    try:
        if max_gt>max_answer:
            multiplyer_ans=max_gt/max_answer
        elif max_gt<max_answer:
            multiplyer_gt = max_answer/max_gt
    except:
        None

    ##TEST
    if True:
        multiplyer_ans = 1
        multiplyer_gt = 1
    for data_j in data_json:
        class_name = 'PIPE'
        
        image_file = data_j[const.JSON_IMG]
        if isinstance(image_file, list):
            image_file=image_file[0]

       
        image_name = image_file[image_file.rfind('/')+1: image_file.rfind('.')]
        #print (temp_dir+"ground-truth/"+image_name+".txt")
        time_now_file = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        ground_file = open(temp_dir+const.GT+"/"+image_name+time_now_file+".txt", 'a')
        res_file = open(temp_dir+const.DET_RES+"/"+image_name+time_now_file+'.txt', 'a')
        
        answer_roi = data_j[const.JSON_ANS]
        gt_roi = data_j[const.JSON_GT]
        if not isinstance(gt_roi, list):
            gt_roi = extract_roi(gt_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')

        if not isinstance(answer_roi, list):
            answer_roi = extract_roi(answer_roi.replace(" ",""), pattern = '<(\d+)><(\d+)><(\d+)><(\d+)>')

        ###calculate IOU
        answer_iou = np.zeros(len(answer_roi))
        for i, pred_box in enumerate(answer_roi):
            iou_threshold = 0
            for j, gt_box in enumerate(gt_roi):
                iou = calculate_iou_hbb(pred_box, gt_box)
                if iou >= iou_threshold:
                    answer_iou[i] = iou
                    iou_threshold = iou

        ###
        
        if analytic:
            if len(gt_roi) >2:
                class_name = class_name+'s'#question_dic['text'][question_dic['text'].index("<p>")+3:question_dic['text'].index("</p>")]
            else:
                class_name = str(len(gt_roi))+'_'+class_name

        HBB = True
        for a_roi, a_iou in zip(answer_roi,answer_iou):
            cx = (int(a_roi[0])+int(a_roi[2]))/2
            cy = (int(a_roi[1])+int(a_roi[3]))/2
            wi = int(a_roi[2])-int(a_roi[0])
            hi = int(a_roi[3])-int(a_roi[1])
            #hbb_box = obb_to_hbb(cx,cy,wi,hi,int(a_roi[4]))
            
            if HBB:
                write_result = class_name+' '+str(a_iou)+' '+str(int(a_roi[0])*multiplyer_ans)+' '+str(int(a_roi[1])*multiplyer_ans)+' '+str(int(a_roi[2])*multiplyer_ans)+' '+str(int(a_roi[3])*multiplyer_ans)
            else:
                write_result = class_name+' '+str(a_iou)+' '+str(int(hbb_box[0])*multiplyer_ans)+' '+str(int(hbb_box[1])*multiplyer_ans)+' '+str(int(hbb_box[2])*multiplyer_ans)+' '+str(int(hbb_box[3])*multiplyer_ans)
                
            res_file.write(write_result+"\n")
            
        for a_roi in gt_roi:
            write_result = class_name+' '+str(int(a_roi[0])*multiplyer_gt)+' '+str(int(a_roi[1])*multiplyer_gt)+' '+str(int(a_roi[2])*multiplyer_gt)+' '+str(int(a_roi[3])*multiplyer_gt)
            ground_file.write(write_result+"\n")
    
        ground_file.close()
        res_file.close()
    return temp_dir+const.GT+"/", temp_dir+const.DET_RES+"/", temp_dir, temp_folder+folder_name

def read_json_result(file_name):
    
    file_json = open(file_name)
    json_obj =  json.load(file_json)
    
    try:
        info_json = json_obj[const.JSON_INFO]
        result_task = info_json[const.JSON_TASK]
        result_model = info_json[const.JSON_MODEL]
        result_dataset = info_json[const.JSON_DATASET]
    except:
        result_task = json_obj[const.JSON_TASK]
        result_model = json_obj[const.JSON_MODEL]
        result_dataset = json_obj[const.JSON_DATASET]
    data_json = json_obj[const.JSON_DATA]
    y_true = []
    y_pred = []
    class_names = []
    isCAP = False
    isList = False
    image_size = []
    if result_task in cap_tags:
        isCAP = True
    elif result_task in seg_tags:
        isList = True
    for data_i in data_json:
        try:
            if isCAP:
                y_true.append(data_i[const.JSON_GT].lower())
                #print ('CAP')
            elif isList:
                y_true.append(data_i[const.JSON_GT])
            else:
                y_true.append(data_i[const.JSON_GT].lower().replace(".",""))
        except:
            if isCAP:
                y_true.append(str(data_i[const.JSON_GT]).lower())
            elif isList:
                y_true.append(data_i[const.JSON_GT])
            else:
                y_true.append(str(data_i[const.JSON_GT]).lower().replace(".","").replace("<",""))
                
        try:
            if isCAP:
                y_pred.append(data_i[const.JSON_ANS].lower())
            elif isList:
                y_pred.append(data_i[const.JSON_ANS])
                image_size.append(data_i[const.JSON_CROP])
            else:
                y_pred.append(data_i[const.JSON_ANS].lower().replace(".","").replace("<",""))

        except:
            if isCAP:
                y_pred.append(str(data_i[const.JSON_ANS]).lower())
            elif isList:
                y_pred.append(data_i[const.JSON_ANS])
                image_size.append(data_i[const.JSON_CROP])
            else:
                y_pred.append(str(data_i[const.JSON_ANS]).lower().replace(".","").replace("<",""))

        try:
            if data_i[const.JSON_GT].lower().replace(".","") not in class_names:
                class_names.append(data_i[const.JSON_GT].lower().replace(".",""))
        except:
            continue
    
    if result_task in cls_tags:
        
        accuracy, precision_macro, recall_macro, f1_macro, micro = evaluate_classification(y_true, y_pred, class_names)
        result_dic = {const.JSON_SAVE_INFO:{const.JSON_SAVE_TASK:result_task,
                              const.JSON_SAVE_MODEL:result_model,
                              const.JSON_SAVE_DATA:result_dataset},
                     const.JSON_SAVE_RES:{const.JSON_SAVE_ACC:accuracy,
                               const.JSON_SAVE_PREC:precision_macro,
                               const.JSON_SAVE_RECALL:recall_macro,
                               const.JSON_SAVE_F1:f1_macro,
                               const.JSON_SAVE_MICRO:micro}}
        
        
    elif result_task in cap_tags:

        bleu, metero_score, rouge_score, rouge_l, CIDEr = evaluate_captioning(y_true, y_pred)
        
        result_dic = {const.JSON_SAVE_INFO:{const.JSON_SAVE_TASK:result_task,
                              const.JSON_SAVE_MODEL:result_model,
                              const.JSON_SAVE_DATA:result_dataset},
                     const.JSON_SAVE_RES:{const.JSON_SAVE_BELU :bleu,
                               const.JSON_SAVE_METERO:metero_score,
                               const.JSON_SAVE_ROUGE:rouge_score,
                               const.JSON_SAVE_ROUGEL:rouge_l,
                               const.JSON_SAVE_CIDER:CIDEr
                               }}


    elif result_task in seg_tags:
        m_dice, m_iou, m_vc8, m_vc16 = evaluate_segmentation(y_true, y_pred, image_size)
        result_dic = {const.JSON_SAVE_INFO:{const.JSON_SAVE_TASK:result_task,
                              const.JSON_SAVE_MODEL:result_model,
                              const.JSON_SAVE_DATA:result_dataset},
                     const.JSON_SAVE_RES:{const.JSON_SAVE_MDICE:m_dice,
                                const.JSON_SAVE_MIOU:m_iou,
                                #'mvc8':m_vc8,
                                #'mvc16':m_vc16
                               }}
        
    elif result_task in det_tags:
        isOBB = False
        isBox = False
        pre_box = False
        if "obb" in result_task.lower():
            isOBB = True

        answer_roi = data_json[0][const.JSON_ANS]
        gt_roi = data_json[0][const.JSON_GT]
        if not isinstance(gt_roi, list):  
            isBox = True
            
        if not isinstance(answer_roi, list):  
            pre_box = True
        detection_result = {}
        iou_res = {}
        if isOBB:
            detection_result = evaluate_detection2(data_json, pre_box=pre_box, is_box=isBox, is_obb=isOBB)
        elif "vg" in result_task.lower():# or True:
            #print ("VG")
            detection_result = evaluate_detection2(data_json, pre_box=pre_box, is_box=isBox, is_obb=isOBB)
        
        else:
            # ##Detection v. 1
            print ("HBB",file_name)
            gt_roi_folder, res_roi_folder, temp_save_folder, master_folder = prepare_detection_format(file_name)
            
            ious = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            
            for iou in ious:
                print (f'IoU@{iou}')
                det_map = evaluate_detection(GT_PATH = gt_roi_folder, 
                           DR_PATH=res_roi_folder, 
                           TEMP_FILES_PATH = temp_save_folder+"temps", 
                           output_files_path= temp_save_folder+"output_detection/",  
                           iou=iou,
                           draw_plot = False)
                iou_res[const.JSON_SAVE_AP+str(iou)] = det_map

            #delete old data
            if os.path.exists(gt_roi_folder):
                shutil.rmtree(gt_roi_folder)
            if os.path.exists(res_roi_folder):
                shutil.rmtree(res_roi_folder)
            if os.path.exists(temp_save_folder):
                shutil.rmtree(temp_save_folder)
            if os.path.exists(master_folder):
                shutil.rmtree(master_folder)
        
        # ## end of detection V.1
        result_dic = {const.JSON_SAVE_INFO:{const.JSON_SAVE_TASK:result_task,
                              const.JSON_SAVE_MODEL:result_model,
                              const.JSON_SAVE_DATA:result_dataset},
                     const.JSON_SAVE_RES:{**detection_result, **iou_res}}#detection_result.update(iou_res)}#{'Evaluation of detection task currently under maintenance. Please contact Mr.Pipe'}}

    elif result_task in vqa_tags:
        if result_dataset in disaster_tags:
            result_dic = eva_disaster(file_name,result_dataset)
        else:
            accuracy, precision_macro, recall_macro, f1_macro, micro = evaluate_classification(y_true, y_pred, class_names)
            accuracy, precision, recall, f1 = evaluate_vqa(y_true, y_pred)
    
            result_dic = {const.JSON_SAVE_INFO:{const.JSON_SAVE_TASK:result_task,
                                  const.JSON_SAVE_MODEL:result_model,
                                  const.JSON_SAVE_DATA:result_dataset},
                         const.JSON_SAVE_RES:{const.JSON_SAVE_ACC:accuracy,
                                    #'precision':precision,
                                    #'recall':recall,
                                   const.JSON_SAVE_F1:f1
                                   }}
    else:
        print ('error', result_task)
        result_dic = {const.JSON_SAVE_INFO:{const.JSON_SAVE_TASK:result_task,
                              const.JSON_SAVE_MODEL:result_model,
                              const.JSON_SAVE_DATA:result_dataset}}

    return result_dic

