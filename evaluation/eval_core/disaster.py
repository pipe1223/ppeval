import json
# import evaluation_master as eva_mas
import pandas as pd
import numpy as np
import re
import tianhuieval.classification as cls_eva
import tianhuieval.captioning as cap_eva
from sklearn.preprocessing import MultiLabelBinarizer

def json_loader(json_path):
    file_json = open(json_path)
    json_obj =  json.load(file_json)
    
    try:
        info_json = json_obj['info']
        result_task = info_json['task']
        result_model = info_json['model']
        result_dataset = info_json['dataset']
    except:
        result_task = json_obj['task']
        result_model = json_obj['model']
        result_dataset = json_obj['dataset']
    data_json = json_obj['data']
    
    df = pd.DataFrame(data_json)

    return df, result_task, result_model, result_dataset
    
def fire_disaster(json_path = "/datadisk/evaluation/results/disaster/D3-epoch_5/[IMG_VQA]Fire_disaster_test.json", 
                  save_result_folder_dir = '/datadisk3/evaluation/results/exp/disaster/',
                  model_name = ''):
    
    df, result_task, result_model, result_dataset = json_loader(json_path)
    
    conditions = [
        df['question'].str.contains('What color is the smoke', case=False),  # Color questions
        df['question'].str.contains('What shape is the smoke|diffusion direction', case=False),  # Shape/direction
        df['question'].str.contains('landforms', case=False)  # Landforms
    ]
    
    categories = ['color', 'shape', 'landforms']  # Corresponding categories
    
    df['qtype'] = np.select(conditions, categories, default='other')
    
    color_data = df[df['qtype'] == 'color'].reset_index()
    landforms_data = df[df['qtype'] == 'landforms'].reset_index()
    shape_data = df[df['qtype'] == 'shape'].reset_index()
    
    #COLOR
    
    pattern = r'appears ([a-zA-Z\s-]+)\.'  # Matches "appears {color}."
    color_data['class_answer'] = color_data['answer'].str.extract(pattern)
    
    color_data['class_gt'] = color_data['gt'].str.extract(pattern)
    
    
    unique_class_gt = color_data['class_gt'].unique().tolist()
    class_answer_list = color_data['class_answer'].tolist()
    class_gt_list = color_data['class_gt'].tolist()
    
    print ('FIRE COLOR')
    accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color)
    
    #shape
    pattern = (
        r'appears to have a ([a-zA-Z\s-]+) pattern\. '  # Matches {shape}
        r'The direction of smoke dispersion is ([a-zA-Z\s-]+)\.'  # Matches {direction}
    )
    
    extracted = shape_data['answer'].str.extract(pattern)
    shape_data['class_shape'] = extracted[0]  # First group ({shape})
    shape_data['class_direction'] = extracted[1]  # Second group ({direction})
    
    extracted = shape_data['gt'].str.extract(pattern)
    shape_data['class_shape_gt'] = extracted[0]  # First group ({shape})
    shape_data['class_direction_gt'] = extracted[1]  # Second group ({direction})
    
    #SHAPE
    unique_class_gt = shape_data['class_shape_gt'].unique().tolist()
    class_answer_list = shape_data['class_shape'].tolist()
    class_gt_list = shape_data['class_shape_gt'].tolist()
    
    print ('FIRE SHAPE')
    accuracy_shape, precision_shape, recall_shape, f1_shape, micro_metrics_shape = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_shape, precision_shape, recall_shape, f1_shape, micro_metrics_shape)
    
    #Direction
    unique_class_gt = shape_data['class_direction_gt'].unique().tolist()
    class_answer_list = shape_data['class_direction'].tolist()
    class_gt_list = shape_data['class_direction_gt'].tolist()
    
    print ('FIRE DIRECTION')
    print (cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt))
    accuracy_direction, precision_direction, recall_direction, f1_direction, micro_metrics_direction = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_direction, precision_direction, recall_direction, f1_direction, micro_metrics_direction)
    
    #land 
    
    pattern = r'(?:^|\. )([^:]+):'
    
    landforms_data['class_land'] = landforms_data['answer'].str.findall(pattern)
    landforms_data['class_land'] = landforms_data['class_land'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    landforms_data['class_land_gt'] = landforms_data['gt'].str.findall(pattern)
    landforms_data['class_land_gt'] = landforms_data['class_land_gt'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    #Landforms
    flattened_landforms = landforms_data['class_land_gt'].explode()
    unique_class_gt = flattened_landforms.dropna().unique().tolist()
    
    class_answer_list = landforms_data['class_land'].tolist()
    class_gt_list = landforms_data['class_land_gt'].tolist()
    
    mlb = MultiLabelBinarizer(classes=unique_class_gt)
    
    # Transform ground truth and predictions
    y_true_bin = mlb.fit_transform(class_gt_list)
    y_pred_bin = mlb.transform(class_answer_list)
    
    # Now pass these to your evaluation function
    print("FIRE Landforms (multi_classes)")

    accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land = cls_eva.multi_label(y_true_bin, y_pred_bin, unique_class_gt)
    
    print (accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land)
    
    
    caption_answer_list = landforms_data['answer'].tolist()
    caption_gt_list = landforms_data['gt'].tolist()
    
    print("FIRE Landforms (caption)")

    bleu, metero_score, rouge_score, rouge_l, CIDEr = cap_eva.evaluate_captioning(caption_gt_list, caption_answer_list)
    
    print (bleu, metero_score, rouge_score, rouge_l, CIDEr)

    #create and save json format
    result_dic = {'info':{'task':result_task,
                      'model':result_model,
                      'dataset':result_dataset},
                  'results':{
             'color_results':{'Accuracy':accuracy_color,
               'Precision':precision_color,
               'Recall':recall_color,
               'F1':f1_color,
               'micro':micro_metrics_color},
             'shape_results':{'Accuracy':accuracy_shape,
               'Precision':precision_shape,
               'Recall':recall_shape,
               'F1':f1_shape,
               'micro':micro_metrics_shape},
             'direction_results':{'Accuracy':accuracy_direction,
               'Precision':precision_direction,
               'Recall':recall_direction,
               'F1':f1_direction,
               'micro':micro_metrics_direction},
             'landform_results':{'Accuracy':accuracy_land,
               'Precision':precision_land,
               'Recall':recall_land,
               'F1':f1_land,
               'micro':micro_metrics_land,
                'BELU':bleu,
               'METERO':metero_score,
               'ROUGE':rouge_score,
                'ROUGE-L':rouge_l,
                'CIDEr':CIDEr}}}
    # #save json

    # final_json = json.dumps(result_dic, indent=4)
    # file_name = json_path.split('/')[-1]
    # with open(save_result_folder_dir+model_name+'/'+'['+model_name+"]"+file_name, "w") as outfile:
    #     outfile.write(final_json)

    return result_dic

def flood_disaster(json_path = "/datadisk/evaluation/results/disaster/D3-epoch_5/[IMG_VQA]Flood_disaster_test.json", 
                  save_result_folder_dir = '/datadisk3/evaluation/results/exp/disaster/',
                  model_name = ''):
    
    df, result_task, result_model, result_dataset = json_loader(json_path)
    
    conditions = [
        df['question'].str.contains('What color is ', case=False),  # Color questions
        df['question'].str.contains('What shape is ', case=False),  # Shape/direction
        df['question'].str.contains('landforms', case=False)  # Landforms
    ]
    
    categories = ['color', 'shape', 'landforms']  # Corresponding categories
    
    df['qtype'] = np.select(conditions, categories, default='other')
    
    color_data = df[df['qtype'] == 'color'].reset_index()
    landforms_data = df[df['qtype'] == 'landforms'].reset_index()
    shape_data = df[df['qtype'] == 'shape'].reset_index()
    
    #COLOR
    
    pattern = r'picture is ([a-zA-Z\s-]+)\.'  # Matches "appears {color}."
    color_data['class_answer'] = color_data['answer'].str.extract(pattern)
    
    color_data['class_gt'] = color_data['gt'].str.extract(pattern)
    unique_class_gt = color_data['class_gt'].unique().tolist()
    class_answer_list = color_data['class_answer'].tolist()
    class_gt_list = color_data['class_gt'].tolist()
    
    
    print ('Flood COLOR')
    accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color)
    #land 
    
    pattern = r'(?:^|\. )([^:]+):'
    
    landforms_data['class_land'] = landforms_data['answer'].str.findall(pattern)
    landforms_data['class_land'] = landforms_data['class_land'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    landforms_data['class_land_gt'] = landforms_data['gt'].str.findall(pattern)
    landforms_data['class_land_gt'] = landforms_data['class_land_gt'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    #Land
    flattened_landforms = landforms_data['class_land_gt'].explode()
    unique_class_gt = flattened_landforms.dropna().unique().tolist()
    
    class_answer_list = landforms_data['class_land'].tolist()
    class_gt_list = landforms_data['class_land_gt'].tolist()
    
    mlb = MultiLabelBinarizer(classes=unique_class_gt)
    
    # Transform ground truth and predictions
    y_true_bin = mlb.fit_transform(class_gt_list)
    y_pred_bin = mlb.transform(class_answer_list)
    
    # Now pass these to your evaluation function
    print("Flood Landforms (multi_classes)")
    accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land = cls_eva.multi_label(y_true_bin, y_pred_bin, unique_class_gt)
    
    print (accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land)
    
    caption_answer_list = landforms_data['answer'].tolist()
    caption_gt_list = landforms_data['gt'].tolist()
    
    print("Flood Landforms (caption)")
    bleu, metero_score, rouge_score, rouge_l, CIDEr = cap_eva.evaluate_captioning(caption_gt_list, caption_answer_list)
    
    print (bleu, metero_score, rouge_score, rouge_l, CIDEr)

    #create and save json format
    result_dic = {'info':{'task':result_task,
                      'model':result_model,
                      'dataset':result_dataset},
                  'results':{
             'color_results':{'Accuracy':accuracy_color,
               'Precision':precision_color,
               'Recall':recall_color,
               'F1':f1_color,
               'micro':micro_metrics_color},
             'landform_results':{'Accuracy':accuracy_land,
               'Precision':precision_land,
               'Recall':recall_land,
               'F1':f1_land,
               'micro':micro_metrics_land,
                'BELU':bleu,
               'METERO':metero_score,
               'ROUGE':rouge_score,
                'ROUGE-L':rouge_l,
                'CIDEr':CIDEr}}}
    # #save json

    # final_json = json.dumps(result_dic, indent=4)
    # file_name = json_path.split('/')[-1]
    # with open(save_result_folder_dir+model_name+'/'+'['+model_name+"]"+file_name, "w") as outfile:
    #     outfile.write(final_json)

    return result_dic


    
def landslide_disaster(json_path = "/home/datadisk/evaluation/results/disaster/D2-epoch_5/[IMG_VQA]Landslide_disaster_test.json", 
                  save_result_folder_dir = '/datadisk3/evaluation/results/exp/disaster/',
                  model_name = ''):
    
    df, result_task, result_model, result_dataset = json_loader(json_path)
    
    conditions = [
        df['question'].str.contains('What color is ', case=False),  # Color questions
        df['question'].str.contains('What shape is ', case=False),  # Shape/direction
        df['question'].str.contains('landforms', case=False)  # Landforms
    ]
    
    categories = ['color', 'shape', 'landforms']  # Corresponding categories
    
    df['qtype'] = np.select(conditions, categories, default='other')
    
    color_data = df[df['qtype'] == 'color'].reset_index()
    landforms_data = df[df['qtype'] == 'landforms'].reset_index()
    shape_data = df[df['qtype'] == 'shape'].reset_index()
    
    
    #COLOR
    
    pattern = r'image exhibits a ([a-zA-Z\s-]+)tone\.'  # Matches "appears {color}."
    color_data['class_answer'] = color_data['answer'].str.extract(pattern)
    
    color_data['class_gt'] = color_data['gt'].str.extract(pattern)
    
    
    unique_class_gt = color_data['class_gt'].unique().tolist()
    class_answer_list = color_data['class_answer'].tolist()
    class_gt_list = color_data['class_gt'].tolist()
    
    print ('Landslide COLOR')

    accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color)
    
    
    #Distribution
    pattern = (
        r'image exhibits a ([a-zA-Z\s-]+) distribution\, '  # Matches {shape}
    )
    
    extracted = shape_data['answer'].str.extract(pattern)
    shape_data['class_shape'] = extracted[0]  # First group ({shape})
    
    extracted = shape_data['gt'].str.extract(pattern)
    shape_data['class_shape_gt'] = extracted[0]  # First group ({shape})
    
    #Distribution
    unique_class_gt = shape_data['class_shape_gt'].unique().tolist()
    class_answer_list = shape_data['class_shape'].tolist()
    class_gt_list = shape_data['class_shape_gt'].tolist()
    
    
    print ('Landslide  Distribution')
    print (cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt))

    accuracy_dist, precision_dist, recall_dist, f1_dist, micro_metrics_dist = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_dist, precision_dist, recall_dist, f1_dist, micro_metrics_dist)
    
    
    #land 
    
    pattern = r'(?:^|\. )([^:]+):'
    
    landforms_data['class_land'] = landforms_data['answer'].str.findall(pattern)
    landforms_data['class_land'] = landforms_data['class_land'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    landforms_data['class_land_gt'] = landforms_data['gt'].str.findall(pattern)
    landforms_data['class_land_gt'] = landforms_data['class_land_gt'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    #Direction
    flattened_landforms = landforms_data['class_land_gt'].explode()
    unique_class_gt = flattened_landforms.dropna().unique().tolist()
    
    class_answer_list = landforms_data['class_land'].tolist()
    class_gt_list = landforms_data['class_land_gt'].tolist()
    
    mlb = MultiLabelBinarizer(classes=unique_class_gt)
    
    # Transform ground truth and predictions
    y_true_bin = mlb.fit_transform(class_gt_list)
    y_pred_bin = mlb.transform(class_answer_list)
    
    # Now pass these to your evaluation function
    print("Landslide Landforms (multi_classes)")

    accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land = cls_eva.multi_label(y_true_bin, y_pred_bin, unique_class_gt)
    
    print (accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land)
    
    
    caption_answer_list = landforms_data['answer'].tolist()
    caption_gt_list = landforms_data['gt'].tolist()
    
    print("Landslide Landforms (caption)")
    bleu, metero_score, rouge_score, rouge_l, CIDEr = cap_eva.evaluate_captioning(caption_gt_list, caption_answer_list)
    
    print (bleu, metero_score, rouge_score, rouge_l, CIDEr)

    #create and save json format
    result_dic = {'info':{'task':result_task,
                      'model':result_model,
                      'dataset':result_dataset},
                  'results':{
             'color_results':{'Accuracy':accuracy_color,
               'Precision':precision_color,
               'Recall':recall_color,
               'F1':f1_color,
               'micro':micro_metrics_color},
             'dist_results':{'Accuracy':accuracy_dist,
               'Precision':precision_dist,
               'Recall':recall_dist,
               'F1':f1_dist,
               'micro':micro_metrics_dist},
             'landform_results':{'Accuracy':accuracy_land,
               'Precision':precision_land,
               'Recall':recall_land,
               'F1':f1_land,
               'micro':micro_metrics_land,
                'BELU':bleu,
               'METERO':metero_score,
               'ROUGE':rouge_score,
                'ROUGE-L':rouge_l,
                'CIDEr':CIDEr}}}
    # #save json

    # final_json = json.dumps(result_dic, indent=4)
    # file_name = json_path.split('/')[-1]
    # with open(save_result_folder_dir+model_name+'/'+'['+model_name+"]"+file_name, "w") as outfile:
    #     outfile.write(final_json)

    return result_dic

def volcano_disaster(json_path = "/datadisk/evaluation/results/disaster/D3-epoch_5/[IMG_VQA]Volcano_disaster_test.json", 
                  save_result_folder_dir = '/datadisk3/evaluation/results/exp/disaster/',
                  model_name = ''):
    
    df, result_task, result_model, result_dataset = json_loader(json_path)
    
    conditions = [
        df['question'].str.contains('What color is', case=False),  # Color questions
        df['question'].str.contains('What shape is', case=False),  # Shape/direction
        df['question'].str.contains('landforms', case=False)  # Landforms
    ]
    
    categories = ['color', 'shape', 'landforms']  # Corresponding categories
    
    df['qtype'] = np.select(conditions, categories, default='other')
    
    color_data = df[df['qtype'] == 'color'].reset_index()
    landforms_data = df[df['qtype'] == 'landforms'].reset_index()
    shape_data = df[df['qtype'] == 'shape'].reset_index()
        
    pattern = r'this picture is ([a-zA-Z\s-]+)\.'  # Matches "appears {color}."
    color_data['class_color'] = color_data['answer'].str.extract(pattern)
    
    color_data['class_color_gt'] = color_data['gt'].str.extract(pattern)
    
    pattern = r'The surface of the mountain is covered with ([a-zA-Z\s-]+)\.'  # Matches "appears {color}."
    color_data['class_covered'] = color_data['answer'].str.extract(pattern)
    
    color_data['class_covered_gt'] = color_data['gt'].str.extract(pattern)
    color_data['class_covered_gt'] = color_data['class_covered_gt'].fillna('No')
    
    
    #color
    unique_class_gt = color_data['class_color_gt'].unique().tolist()
    class_answer_list = color_data['class_color'].tolist()
    class_gt_list = color_data['class_color_gt'].tolist()
    
    print ('Volcano Color')
    accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_color, precision_color, recall_color, f1_color, micro_metrics_color)
    
    #covered
    unique_class_gt = color_data['class_covered_gt'].unique().tolist()
    class_answer_list = color_data['class_covered'].tolist()
    class_gt_list = color_data['class_covered_gt'].tolist()
    
    print ('Volcano covered')
    accuracy_covered, precision_covered, recall_covered, f1_covered, micro_metrics_covered = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_covered, precision_covered, recall_covered, f1_covered, micro_metrics_covered)


    #SHAPE
    
    pattern = r'conical in shape, ([a-zA-Z\s-]+)\ textures on'  # Matches "appears {color}."
    shape_data['class_answer'] = shape_data['answer'].str.extract(pattern)
    
    shape_data['class_gt'] = shape_data['gt'].str.extract(pattern)
    
    
    unique_class_gt = shape_data['class_gt'].unique().tolist()
    class_answer_list = shape_data['class_answer'].tolist()
    class_gt_list = shape_data['class_gt'].tolist()
    
    print ('Vocalno SHAPE')
    accuracy_shape, precision_shape, recall_shape, f1_shape, micro_metrics_shape = cls_eva.single_label(class_gt_list,class_answer_list,unique_class_gt)
    
    print (accuracy_shape, precision_shape, recall_shape, f1_shape, micro_metrics_shape)


    #land 
    
    pattern = r'(?:^|\. )([^:]+):'
    
    landforms_data['class_land'] = landforms_data['answer'].str.findall(pattern)
    landforms_data['class_land'] = landforms_data['class_land'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    landforms_data['class_land_gt'] = landforms_data['gt'].str.findall(pattern)
    landforms_data['class_land_gt'] = landforms_data['class_land_gt'].apply(
        lambda x: [item.strip() for item in x]  # Remove leading/trailing whitespace
    )
    
    
    flattened_landforms = landforms_data['class_land_gt'].explode()
    unique_class_gt = flattened_landforms.dropna().unique().tolist()
    
    class_answer_list = landforms_data['class_land'].tolist()
    class_gt_list = landforms_data['class_land_gt'].tolist()
    
    mlb = MultiLabelBinarizer(classes=unique_class_gt)
    
    # Transform ground truth and predictions
    y_true_bin = mlb.fit_transform(class_gt_list)
    y_pred_bin = mlb.transform(class_answer_list)
    
    # Now pass these to your evaluation function
    print("Vocalno Landforms (multi_classes)")
    accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land = cls_eva.multi_label(y_true_bin, y_pred_bin, unique_class_gt)
    
    print (accuracy_land, precision_land, recall_land, f1_land, micro_metrics_land)
    
    caption_answer_list = landforms_data['answer'].tolist()
    caption_gt_list = landforms_data['gt'].tolist()
    
    print("Vocalno Landforms (caption)")
    bleu, metero_score, rouge_score, rouge_l, CIDEr = cap_eva.evaluate_captioning(caption_gt_list, caption_answer_list)
    
    print (bleu, metero_score, rouge_score, rouge_l, CIDEr)

    
    #create and save json format
    result_dic = {'info':{'task':result_task,
                      'model':result_model,
                      'dataset':result_dataset},
                  'results':{
             'color_results':{'Accuracy':accuracy_color,
               'Precision':precision_color,
               'Recall':recall_color,
               'F1':f1_color,
               'micro':micro_metrics_color},
            'shape_results':{'Accuracy':accuracy_shape,
               'Precision':precision_shape,
               'Recall':recall_shape,
               'F1':f1_shape,
               'micro':micro_metrics_shape},
             'covered_results':{'Accuracy':accuracy_covered,
               'Precision':precision_covered,
               'Recall':recall_covered,
               'F1':f1_covered,
               'micro':micro_metrics_covered},
              'landform_results':{'Accuracy':accuracy_land,
               'Precision':precision_land,
               'Recall':recall_land,
               'F1':f1_land,
               'micro':micro_metrics_land,
                'BELU':bleu,
               'METERO':metero_score,
               'ROUGE':rouge_score,
                'ROUGE-L':rouge_l,
                'CIDEr':CIDEr}}}
    # #save json

    # final_json = json.dumps(result_dic, indent=4)
    # file_name = json_path.split('/')[-1]
    # with open(save_result_folder_dir+model_name+'/'+'['+model_name+"]"+file_name, "w") as outfile:
    #     outfile.write(final_json)
    return result_dic

def main(args):
    file_json = open(json_path)
    json_obj =  json.load(file_json)

    info_json = json_obj['info']
    result_task = info_json['task']
    result_model = info_json['model']
    result_dataset = info_json['dataset']


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--evaluation-file", type=str, default="")
#     parser.add_argument("--evaluation-folder", type=str, default="")
#     parser.add_argument("--model_name", type=str, default="")
#     parser.add_argument("--isReplace", type=bool, default=False)
#     parser.add_argument("--save_path", type=str, default='/home/datadisk3/evaluation/results/20241010_test/')
#     parser.add_argument("--debug", type=str, default='no')
#     args = parser.parse_args()

#     main(args)

def eva_disaster(json_path, disaster_type):
    if disaster_type == 'Volcano':
        result_dic = volcano_disaster(json_path)
    if disaster_type == 'Flood':
        result_dic = flood_disaster(json_path)
    if disaster_type == 'Fire': 
        result_dic = fire_disaster(json_path)
    if disaster_type == 'landslide': 
        result_dic = landslide_disaster(json_path)

    return result_dic
