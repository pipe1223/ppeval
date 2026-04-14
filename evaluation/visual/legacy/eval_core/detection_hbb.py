import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import matplotlib.pyplot as plt

import numpy as np


def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])


    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            color,
            lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # 
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.manager.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height 
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
def evaluate_detection_from_dir(GT_PATH = 'tmp/ground-truth/',
                       DR_PATH = 'tmp/detection-results/',
                       TEMP_FILES_PATH = "tmp/mAP/temp_files",
                       output_files_path = "tmp/mAP/output",
                       iou =0.5,
                      draw_plot = False):
    MINOVERLAP = iou
    specific_iou_flagged = False 
    no_animation = True #can remove
    show_animation = False
    
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    
    if os.path.exists(output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)

    os.makedirs(output_files_path)
    if draw_plot:
        os.makedirs(os.path.join(output_files_path, "classes"))
   

    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    
    for txt_file in ground_truth_files_list:
        #print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
    
                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)
            
    
        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    print (gt_counter_per_class)
    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)



    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            #print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    #print("match")
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                    #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    sum_prec = 0.0  # <-- Add this
    sum_rec = 0.0   # <-- Add this
    sum_f1 = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    
    # open file to store the output
    total_nd = 0
    total_tp = 0
    total_fp = 0
    
    # --- NEW CODE: Add these dictionaries to store overall metrics ---
    max_f1_dictionary = {}
    class_precision = {}
    class_recall = {}
    
    # --- NEW CODE END ---
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        
        for class_index, class_name in enumerate(gt_classes):
            print (class_name)
            count_true_positives[class_name] = 0
            """
             Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            
            #print (dr_data)
            """
             Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            total_nd += nd
            
            # --- NEW CODE: Save the original TP/FP arrays before cumulative sum ---
            tp_original = [0] * nd   # This will hold 1 for TP, 0 for not
            fp_original = [0] * nd   # This will hold 1 for FP, 0 for not
            
            # --- NEW CODE END ---
            tp = [0] * nd # creates an array of zeros of size nd
            iou_tester = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                
                # open ground-truth with that file_id
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ]
                
                #old
                # for obj in ground_truth_data:
                #     # look for a class_name match
                #     if obj["class_name"] == class_name:
                #         bbgt = [ float(x) for x in obj["bbox"].split() ]
                #         bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                #         iw = bi[2] - bi[0] + 1
                #         ih = bi[3] - bi[1] + 1
                #         if iw > 0 and ih > 0:
                #             # compute overlap (IoU) = area of intersection / area of union
                #             ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                #                             + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                #             ov = iw * ih / ua
                #             if ov > ovmax:
                #                 ovmax = ov
                #                 gt_match = obj
                
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        
                        # Calculate intersection coordinates
                        x1 = max(bb[0], bbgt[0])
                        y1 = max(bb[1], bbgt[1])
                        x2 = min(bb[2], bbgt[2])
                        y2 = min(bb[3], bbgt[3])
                        
                        # Calculate intersection area
                        intersection_width = max(0, x2 - x1 + 1)  # +1 for inclusive
                        intersection_height = max(0, y2 - y1 + 1)  # +1 for inclusive
                        intersection_area = intersection_width * intersection_height
                        
                        if intersection_area > 0:
                            # Calculate areas of both boxes
                            area_bb = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                            area_bbgt = (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                            
                            # Calculate union area
                            union_area = area_bb + area_bbgt - intersection_area
                            
                            # Calculate IoU
                            iou = intersection_area / union_area
                            
                            if iou > ovmax:
                                ovmax = iou
                                gt_match = obj
                            print ("HBB iou:", ovmax)
                
                # set minimum overlap
                min_overlap = MINOVERLAP
                
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        print ("not difficult")
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            tp_original[idx] = 1 # --- NEW CODE ---
                            iou_tester[idx]=ovmax
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                        
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                            fp_original[idx] = 1 # --- NEW CODE ---
                
                else:
                    # false positive
                    fp[idx] = 1
                    fp_original[idx] = 1 # --- NEW CODE ---
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"
                
                """
                 Draw image to show animation
                """
            
            print(tp)
            print ('aaa',iou_tester)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val 
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            print("RECALL:",rec)
            prec = tp[:]          
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            
            print(prec)
            
            # --- NEW CODE START: Calculate CORRECT Precision/Recall/F1 ---
            # Calculate cumulative sums CORRECTLY for the original TP/FP arrays
            # We need to sort the detections by confidence first. 
            # ASSUMPTION: dr_data is already sorted by confidence descending.
            # If it's not, you must sort dr_data, tp_original, and fp_original here.
            
            cumulative_tp = np.cumsum(tp_original).tolist()
            cumulative_fp = np.cumsum(fp_original).tolist()
            
            correct_precisions = []
            correct_recalls = []
            correct_f1_scores = []
            
            for i in range(nd):
                # Precision = TP / (TP + FP) = Cumulative TP / Total predictions so far
                total_predictions_so_far = i + 1
                precision_i = cumulative_tp[i] / total_predictions_so_far if total_predictions_so_far > 0 else 0
                correct_precisions.append(precision_i)
                
                # Recall = TP / All GT = Cumulative TP / Total ground truths
                recall_i = cumulative_tp[i] / gt_counter_per_class[class_name] if gt_counter_per_class[class_name] > 0 else 0
                correct_recalls.append(recall_i)
                
                # F1 Score
                if (precision_i + recall_i) > 0:
                    f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
                else:
                    f1_i = 0.0
                correct_f1_scores.append(f1_i)
            
            # Find the point with the maximum F1 score
            max_f1_index = np.argmax(correct_f1_scores)
            overall_precision = correct_precisions[max_f1_index]
            overall_recall = correct_recalls[max_f1_index]
            max_f1 = correct_f1_scores[max_f1_index]
            
            # Store for this class
            max_f1_dictionary[class_name] = max_f1
            class_precision[class_name] = overall_precision
            class_recall[class_name] = overall_recall
            
            # Sum for final macro-average
            sum_prec += overall_precision
            sum_rec += overall_recall
            sum_f1 += max_f1
            
            print(f"[CORRECTED] Class: {class_name}")
            print(f"-> Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-Score: {max_f1:.4f}")
            # --- NEW CODE END ---
    
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
            """
             Write to output.txt
            """
            rounded_prec = [ '%.2f' % elem for elem in prec ]
            rounded_rec = [ '%.2f' % elem for elem in rec ]

            
            output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            #print(text)#PIPE
            ap_dictionary[class_name] = ap*100
    
            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
            lamr_dictionary[class_name] = lamr
    
            """
             Draw plot
            """
            #print (lamr)
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf() # gcf - get current figure
                fig.canvas.manager.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                #plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca() # gca - get current axes
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05]) # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                #while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                #plt.show()
                # save the plot
                fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                plt.cla() # clear axes for next plot
    

    
        output_file.write("\n# mAP of all classes\n")

        
        try:
            macro_precision = sum_prec / n_classes
            macro_recall = sum_rec / n_classes
            macro_f1 = sum_f1 / n_classes
            
            mAP = sum_AP / n_classes
            
        except:
            mAP = 0.0
        text = "mAP = {0:.2f}%".format(mAP*100)
        output_file.write(text + "\n")
        #print(text)#PIPE

        # --- NEW CODE: Write the final macro-averaged results ---
        output_file.write("\n# Corrected Overall Metrics (Macro-Averaged at Max F1 point)\n")
        macro_precision = sum_prec / n_classes
        macro_recall = sum_rec / n_classes
        macro_f1 = sum_f1 / n_classes


    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)
    
    """
     Count total of detection-results
    """
    # iterate through all the files
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_name] = 1

    dr_classes = list(det_counter_per_class.keys())


    """
     Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = output_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )
    
    """
     Write number of ground-truth objects per class to results.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            output_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")
    
    """
     Finish counting true positives
    """
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    #print(count_true_positives)

    """
     Plot the total number of occurences of each class in the "detection-results" folder
    """
    if draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(dr_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = output_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
            )

    """
     Write number of detected objects per class to output.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            output_file.write(text)

    """
     Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )
    
    """
     Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = output_files_path + "/mAP.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    return mAP*100, ap_dictionary, gt_counter_per_class, macro_precision*100, macro_recall*100, macro_f1*100

# =============================================================================
# In-memory API (no filesystem I/O)
# =============================================================================

def _hbb_iou_inclusive(bb, bbgt):
    """
    IoU for HBB using the same inclusive (+1) convention as the legacy evaluator.
    bb, bbgt: [left, top, right, bottom]
    """
    x1 = max(bb[0], bbgt[0])
    y1 = max(bb[1], bbgt[1])
    x2 = min(bb[2], bbgt[2])
    y2 = min(bb[3], bbgt[3])

    iw = max(0.0, x2 - x1 + 1.0)
    ih = max(0.0, y2 - y1 + 1.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_bb = (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
    area_gt = (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
    union = area_bb + area_gt - inter
    if union <= 0:
        return 0.0
    return inter / union


def _as_hbb(bbox):
    """
    Normalize bbox to [l, t, r, b] float list.
    Accepts:
      - [l,t,r,b]
      - {'bbox': [...]} or {'coord': [...]} etc handled elsewhere
      - quad [x1,y1,...,x4,y4] -> converted to axis-aligned bounding box (fallback)
    """
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple, np.ndarray)):
        arr = list(bbox)
        if len(arr) == 4:
            l, t, r, b = arr
            return [float(l), float(t), float(r), float(b)]
        if len(arr) == 8:
            xs = [arr[0], arr[2], arr[4], arr[6]]
            ys = [arr[1], arr[3], arr[5], arr[7]]
            return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    raise ValueError(f"Unsupported bbox format for HBB: {bbox!r}")


def _get_field(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _normalize_gt_box(box, default_class_name="PIPE"):
    """
    Normalize one GT box to:
      {'class_name': str, 'bbox': [l,t,r,b], 'used': False, (optional) 'difficult': True}
    """
    if box is None:
        return None
    if isinstance(box, dict):
        cls = _get_field(box, ["class_name", "label", "class", "category", "name"], default_class_name)
        bbox = _get_field(box, ["bbox", "box", "coord", "coords", "xyxy", "hbb"], None)
        if bbox is None and "coord" in box and isinstance(box["coord"], (list, tuple)) and len(box["coord"]) == 4:
            bbox = box["coord"]
        difficult = bool(_get_field(box, ["difficult", "is_difficult"], False))
        out = {"class_name": str(cls), "bbox": _as_hbb(bbox), "used": False}
        if difficult:
            out["difficult"] = True
        return out

    # List/tuple formats
    if isinstance(box, (list, tuple, np.ndarray)):
        arr = list(box)
        # [class, l,t,r,b] or [l,t,r,b]
        if len(arr) == 5 and isinstance(arr[0], str):
            cls = arr[0]
            bbox = arr[1:5]
            return {"class_name": str(cls), "bbox": _as_hbb(bbox), "used": False}
        if len(arr) == 4:
            # no label -> default class
            return {"class_name": str(default_class_name), "bbox": _as_hbb(arr), "used": False}

    raise ValueError(f"Unsupported GT box format: {box!r}")


def _normalize_pred_box(box, default_class_name="PIPE"):
    """
    Normalize one prediction box to:
      {'class_name': str, 'bbox': [l,t,r,b], 'confidence': float}
    """
    if box is None:
        return None
    if isinstance(box, dict):
        cls = _get_field(box, ["class_name", "label", "class", "category", "name"], default_class_name)
        bbox = _get_field(box, ["bbox", "box", "coord", "coords", "xyxy", "hbb"], None)
        conf = _get_field(box, ["confidence", "score", "conf", "prob"], 1.0)
        return {"class_name": str(cls), "bbox": _as_hbb(bbox), "confidence": float(conf)}

    if isinstance(box, (list, tuple, np.ndarray)):
        arr = list(box)
        # [class, conf, l,t,r,b]
        if len(arr) == 6 and isinstance(arr[0], str):
            cls = arr[0]
            conf = arr[1]
            bbox = arr[2:6]
            return {"class_name": str(cls), "bbox": _as_hbb(bbox), "confidence": float(conf)}
        # [conf, l,t,r,b]
        if len(arr) == 5 and not isinstance(arr[0], str):
            conf = arr[0]
            bbox = arr[1:5]
            return {"class_name": str(default_class_name), "bbox": _as_hbb(bbox), "confidence": float(conf)}
        # [class, l,t,r,b] (no conf)
        if len(arr) == 5 and isinstance(arr[0], str):
            cls = arr[0]
            bbox = arr[1:5]
            return {"class_name": str(cls), "bbox": _as_hbb(bbox), "confidence": 1.0}

    raise ValueError(f"Unsupported prediction box format: {box!r}")


def _extract_boxes_from_sample(sample, kind):
    """
    kind: 'gt' or 'pred'
    sample can be:
      - list of boxes
      - dict containing boxes under 'gt'/'pred'/'boxes'/'prediction'
    """
    if sample is None:
        return []
    if isinstance(sample, list):
        return sample
    if isinstance(sample, dict):
        if kind == "gt":
            return _get_field(sample, ["gt", "gts", "boxes", "targets", "annotations"], [])
        else:
            return _get_field(sample, ["pred", "preds", "prediction", "detections", "boxes"], [])
    raise ValueError(f"Unsupported sample format: {type(sample)}")


def evaluate_detection_ytrue_ypred(
    y_true,
    y_pred,
    iou=0.5,
    default_class_name="PIPE",
    verbose=False,
):
    """
    In-memory HBB detection evaluator (no file I/O).

    Inputs (aligned lists, length N):
      y_true[i]: list of GT boxes for sample i
      y_pred[i]: list of predicted boxes for sample i

    Each GT box can be:
      - dict: {'label' or 'class_name': str, 'bbox' or 'coord': [l,t,r,b], optional 'difficult': bool}
      - list: [label, l,t,r,b] or [l,t,r,b] (uses default_class_name)

    Each prediction box can be:
      - dict: {'label'/'class_name': str, 'bbox'/'coord': [l,t,r,b], 'confidence'/'score': float}
      - list: [label, conf, l,t,r,b] or [conf, l,t,r,b] or [label, l,t,r,b]

    Returns (same as legacy evaluate_detection):
      (mAP*100, ap_dictionary, gt_counter_per_class, macro_precision*100, macro_recall*100, macro_f1*100)
    """
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must be provided for in-memory evaluation.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    MINOVERLAP = float(iou)

    # Build per-sample GT and prediction structures
    sample_ids = []
    gt_by_sample = {}
    pred_by_sample = {}

    gt_counter_per_class = {}
    counter_images_per_class = {}

    for i in range(len(y_true)):
        gt_sample = y_true[i]
        pred_sample = y_pred[i]

        # Derive a stable id if present, else index string
        sid = None
        if isinstance(gt_sample, dict):
            sid = _get_field(gt_sample, ["id", "image", "image_id", "name"], None)
        if sid is None and isinstance(pred_sample, dict):
            sid = _get_field(pred_sample, ["id", "image", "image_id", "name"], None)
        if sid is None:
            sid = str(i)

        sample_ids.append(str(sid))

        gt_boxes_raw = _extract_boxes_from_sample(gt_sample, "gt")
        pred_boxes_raw = _extract_boxes_from_sample(pred_sample, "pred")

        gt_boxes = []
        seen_classes_in_image = set()
        for b in gt_boxes_raw:
            gb = _normalize_gt_box(b, default_class_name=default_class_name)
            if gb is None:
                continue
            # reset used flag in case caller reused objects
            gb["used"] = False
            gt_boxes.append(gb)

            # Count only non-difficult (legacy behavior)
            if "difficult" not in gb:
                cls = gb["class_name"]
                gt_counter_per_class[cls] = gt_counter_per_class.get(cls, 0) + 1
                if cls not in seen_classes_in_image:
                    counter_images_per_class[cls] = counter_images_per_class.get(cls, 0) + 1
                    seen_classes_in_image.add(cls)

        pred_boxes = []
        for b in pred_boxes_raw:
            pb = _normalize_pred_box(b, default_class_name=default_class_name)
            if pb is None:
                continue
            pred_boxes.append(pb)

        gt_by_sample[str(sid)] = gt_boxes
        pred_by_sample[str(sid)] = pred_boxes

    gt_classes = sorted(list(gt_counter_per_class.keys()))
    n_classes = len(gt_classes)
    if n_classes == 0:
        # No GT at all: match legacy's safest output
        return 0.0, {}, {}, 0.0, 0.0, 0.0

    sum_AP = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    sum_f1 = 0.0

    ap_dictionary = {}

    # For completeness (not returned, but keeps parity with legacy)
    count_true_positives = {c: 0 for c in gt_classes}

    for class_name in gt_classes:
        # Gather detections of this class across all samples
        dr_data = []
        for sid in sample_ids:
            for pb in pred_by_sample[sid]:
                if pb["class_name"] == class_name:
                    dr_data.append({
                        "confidence": float(pb["confidence"]),
                        "file_id": sid,
                        "bbox": pb["bbox"],
                    })
        dr_data.sort(key=lambda x: float(x["confidence"]), reverse=True)

        nd = len(dr_data)
        tp_original = [0] * nd
        fp_original = [0] * nd

        # Reset used flags for this class only
        for sid in sample_ids:
            for obj in gt_by_sample[sid]:
                if obj["class_name"] == class_name:
                    obj["used"] = False

        for idx, det in enumerate(dr_data):
            sid = det["file_id"]
            bb = det["bbox"]

            gt_objs = [o for o in gt_by_sample[sid] if o["class_name"] == class_name]
            ovmax = -1.0
            gt_match = None

            for obj in gt_objs:
                bbgt = obj["bbox"]
                ov = _hbb_iou_inclusive(bb, bbgt)
                if ov > ovmax:
                    ovmax = ov
                    gt_match = obj

            if ovmax >= MINOVERLAP and gt_match is not None:
                if "difficult" not in gt_match:
                    if not bool(gt_match.get("used", False)):
                        tp_original[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                    else:
                        fp_original[idx] = 1
                # else: difficult -> ignore (tp=0, fp=0), matches legacy behavior
            else:
                fp_original[idx] = 1

        # Precision/Recall arrays for AP (legacy uses cumulative TP/FP where ignored detections don't count)
        tp_cum = np.cumsum(tp_original).astype(float)
        fp_cum = np.cumsum(fp_original).astype(float)

        rec = []
        prec = []
        denom_gt = float(gt_counter_per_class.get(class_name, 0))
        for i_det in range(nd):
            rec_i = (tp_cum[i_det] / denom_gt) if denom_gt > 0 else 0.0
            denom = tp_cum[i_det] + fp_cum[i_det]
            prec_i = (tp_cum[i_det] / denom) if denom > 0 else 0.0
            rec.append(float(rec_i))
            prec.append(float(prec_i))

        # Corrected per-class overall precision/recall/f1 at max F1 point (legacy NEW CODE)
        if nd == 0:
            overall_precision = 0.0
            overall_recall = 0.0
            max_f1 = 0.0
        else:
            cumulative_tp = np.cumsum(tp_original).astype(float)
            correct_precisions = []
            correct_recalls = []
            correct_f1_scores = []
            for i_det in range(nd):
                precision_i = cumulative_tp[i_det] / float(i_det + 1)  # total predictions so far
                recall_i = (cumulative_tp[i_det] / denom_gt) if denom_gt > 0 else 0.0
                if (precision_i + recall_i) > 0:
                    f1_i = 2.0 * (precision_i * recall_i) / (precision_i + recall_i)
                else:
                    f1_i = 0.0
                correct_precisions.append(precision_i)
                correct_recalls.append(recall_i)
                correct_f1_scores.append(f1_i)

            max_f1_index = int(np.argmax(correct_f1_scores))
            overall_precision = float(correct_precisions[max_f1_index])
            overall_recall = float(correct_recalls[max_f1_index])
            max_f1 = float(correct_f1_scores[max_f1_index])

        sum_prec += overall_precision
        sum_rec += overall_recall
        sum_f1 += max_f1

        ap, _, _ = voc_ap(rec[:], prec[:])
        sum_AP += ap
        ap_dictionary[class_name] = float(ap) * 100.0

        if verbose:
            print(f"[HBB] {class_name} AP={ap*100:.2f} P={overall_precision:.4f} R={overall_recall:.4f} F1={max_f1:.4f}")

    mAP = (sum_AP / n_classes) if n_classes > 0 else 0.0
    macro_precision = (sum_prec / n_classes) if n_classes > 0 else 0.0
    macro_recall = (sum_rec / n_classes) if n_classes > 0 else 0.0
    macro_f1 = (sum_f1 / n_classes) if n_classes > 0 else 0.0

    return float(mAP) * 100.0, ap_dictionary, gt_counter_per_class, float(macro_precision) * 100.0, float(macro_recall) * 100.0, float(macro_f1) * 100.0


def evaluate_detection(*args, **kwargs):
    """
    Backward-compatible dispatcher.

    - New usage (in-memory):
        evaluate_detection(y_true, y_pred, iou=0.5, ...)

    - Legacy usage (filesystem):
        evaluate_detection(GT_PATH='...', DR_PATH='...', TEMP_FILES_PATH='...', output_files_path='...', iou=0.5, draw_plot=False)
    """
    # No args/kwargs: preserve legacy defaults
    if len(args) == 0 and len(kwargs) == 0:
        return evaluate_detection_from_dir()

    # If called with legacy keyword args or a string first arg, treat as directory-based
    legacy_keys = {"GT_PATH", "DR_PATH", "TEMP_FILES_PATH", "output_files_path", "draw_plot"}
    if any(k in kwargs for k in legacy_keys) or (len(args) > 0 and isinstance(args[0], str)):
        return evaluate_detection_from_dir(*args, **kwargs)

    # Otherwise treat as in-memory
    return evaluate_detection_ytrue_ypred(*args, **kwargs)
