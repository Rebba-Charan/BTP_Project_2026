import cv2
import numpy as np
import pandas as pd
from detectionTracking import Tracking
from actionClassification import ActionClassifier
from engagementClassification import EngagementClassifier
import math
import utils
import csv
from engageConfig import *





if __name__=='__main__':

    if display_enabled:
        cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
    tracking_display = display_enabled
    action_display_begining_clip =True #== display action labels at the begining of each video clip (3 seconds)

    classes_all = pd.read_csv(labels_file, usecols=['id', 'name'])
    action_names = [class_des for i, class_des in classes_all.values.tolist()]
    #===initialize tracker
    tracking = Tracking(objectDetector='fastercnn',tracker='ByteTrack')
    actionClassifier = ActionClassifier(action_names)
    engagementClassifier = EngagementClassifier(action_names,usedLLM='llama',contextAware=True)

    vid = cv2.VideoCapture(v_path)
    if not vid.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {v_path}. "
            "Set v_path to an existing video in engageConfig.py."
        )
    fps = vid.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError(f"Could not read FPS from video: {v_path}")
    print(f"Processing {v_path} ({fps:.2f} FPS)")

    frm_count =0
    sts_trackers_data ={}
    sts_predActionsSeq ={}
    sts_engagement_levels_dict = {}
    frms =[]
    displayFrms =[]
    predictions = []
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        display_frame = frame.copy()

        frm_count+=1
        frms.append(frame.copy())
        dets = tracking.detect(frame)
        trackingFrmData = tracking.track(dets,frame)
        for t_id,box,state in trackingFrmData:
            if t_id in sts_trackers_data:

                 sts_trackers_data[t_id]['boxes'].append(box)
                 sts_trackers_data[t_id]['currentPosition'] = box
                 sts_trackers_data[t_id]['lstFrmNum'] = frm_count

            else:
                sts_trackers_data[t_id] = {'boxes': [box],'currentPosition':box,'predAction':'','walking_distance':0,'lstFrmNum':frm_count,'state':'active'}
        #===recognize actions every 3 seconds ===#
        if frm_count>=clip_len and frm_count%clip_len==0:
            print(f"=================****************************==============")
            for st_id, bboxes in sts_trackers_data.items():
                boxesLst = bboxes['boxes'][frm_count-clip_len:frm_count]
                if len(boxesLst) < 20:
                    #sts_trackers_data[st_id]['boxes'] = []
                    sts_trackers_data[st_id]['state'] = 'not-active'
                    sts_trackers_data[st_id]['walking_distance'] = 0
                    continue
                min_x,min_y,max_x,max_y = utils.getEnclosingBBox(boxesLst)
                if abs(max_y-min_y)< 100 or abs(max_x-min_x)<100:
                    print(f"Id: {st_id} has min width {min_x}-{max_x} or height {min_y}-{max_y}")
                    continue
                #====make sure if it is within the image boundry===#
                min_x,min_y,max_x,max_y = utils.checkBoundary((min_x,min_y,max_x,max_y), frame.shape[:-1])

                subFrms = [frm[min_y:max_y,min_x:max_x] for frm in frms]
                inds = actionClassifier.uniformSampling(len(subFrms))
                sampledSubFrms = [subFrms[i] for i in inds]
                try:
                    pred_action = actionClassifier.classify(sampledSubFrms)
                    start_frame = frm_count - clip_len
                    start_time = start_frame / fps
                    end_time = frm_count / fps
                    print(
                        f" Id: {st_id} [{start_time:.2f}-{end_time:.2f}s] "
                        f"Action: {pred_action}"
                    )
                    predictions.append({
                        "student_id": st_id,
                        "start_frame": start_frame,
                        "end_frame": frm_count,
                        "start_time": round(start_time, 3),
                        "end_time": round(end_time, 3),
                        "predicted_label": pred_action,
                    })
                    #sts_trackers_data[st_id]['boxes'] =[]
                    sts_trackers_data[st_id]['walking_distance'] = 0
                    if st_id in sts_predActionsSeq:
                        sts_predActionsSeq[st_id].append(pred_action)
                    else:
                        sts_predActionsSeq[st_id] =[pred_action]

                    sts_trackers_data[st_id]['predAction']=pred_action
                except Exception as e:
                    print(f"Error: {e}")
            displayFrms = frms.copy()
            frms.clear()

        #===engagement classification===#
        if frm_count % 1800 == 0:
            print(f"////////****************************/////////")
            #====filter sts_predActionsSeq===#
            try:
                sts_engagement_levels_dict = engagementClassifier.classify_action_sequence(sts_predActionsSeq)
            except Exception as e:
                    print(f"Error: {e}")
            sts_predActionsSeq.clear()

        if tracking_display:
            if not action_display_begining_clip:
                for t_id, tracker_data in sts_trackers_data.items():
                    t_last_frm_num = tracker_data['lstFrmNum']
                    # if abs(t_last_frm_num-frm_count) > 5:
                    #     continue
                    if tracker_data['state'] != 'active':
                        continue
                    curPos = tracker_data['currentPosition']
                    # print(curPos)
                    predAction = tracker_data['predAction']
                    x1, y1 = int(curPos[0]), int(curPos[1])

                    display_frame = cv2.putText(display_frame, f"{predAction}", (x1, max(0, y1 - 40)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 100, 200), 3, )
                    if t_id in sts_engagement_levels_dict:
                        display_frame = cv2.putText(display_frame, f"{sts_engagement_levels_dict[t_id]}",
                                                    (x1, max(0, y1 - 80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250),
                                                    3, )
                    # print(frm_count-clip_len)
                    tracking.tracker.plot_results(display_frame, show_trajectories=False)
                    cv2.imshow('Tracking', display_frame)
            else:
                if frm_count>= clip_len:
                    display_frame = displayFrms.pop(0)
                    trackersInfo=[]
                    for t_id, tracker_data in sts_trackers_data.items():
                        t_last_frm_num = tracker_data['lstFrmNum']
                        # if abs(t_last_frm_num-frm_count) > 5:
                        #     continue
                        if tracker_data['state']!= 'active':
                            continue
                        curPos = tracker_data['currentPosition']
                        #print(curPos)
                        predAction = tracker_data['predAction']
                        x1,y1= int(curPos[0]),int(curPos[1])

                        display_frame = cv2.putText(display_frame,f"{predAction}",(x1, max(0, y1 - 40)),cv2.FONT_HERSHEY_SIMPLEX,1,(10,100,200),3,)
                        if t_id in sts_engagement_levels_dict:
                            display_frame = cv2.putText(display_frame,f" {sts_engagement_levels_dict[t_id]}",(x1, max(0, y1 - 80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 3, )
                        #print(frm_count-clip_len)
                        if (frm_count-clip_len)>= len(tracker_data['boxes']):
                            trackersInfo.append((t_id, tracker_data['boxes'][-1]))
                        else:
                         trackersInfo.append((t_id,tracker_data['boxes'][frm_count-clip_len]))
                    # Plot results and display
                    display_frame= tracking.plot_boxes_on_img(display_frame,trackersInfo)
                    #tracking.tracker.plot_results(display_frame, show_trajectories=False)
                    cv2.imshow('Tracking', display_frame)
                # cv2.namedWindow('Frms', cv2.WINDOW_NORMAL)
                # cv2.imshow('Frms', frms[-1])

            if display_enabled and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    vid.release()
    with open(multi_predictions_file, "w", newline="", encoding="utf-8") as output:
        fields = [
            "student_id", "start_frame", "end_frame",
            "start_time", "end_time", "predicted_label"
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Saved {len(predictions)} predictions to {multi_predictions_file}")
    if display_enabled:
        cv2.destroyAllWindows()
