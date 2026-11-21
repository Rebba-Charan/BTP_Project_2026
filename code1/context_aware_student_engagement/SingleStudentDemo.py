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

if __name__ == '__main__':

    if display_enabled:
        cv2.namedWindow('Student Engagement Measurement Demo', cv2.WINDOW_NORMAL)
    st_id =0

    classes_all = pd.read_csv(labels_file, usecols=['id', 'name'])
    action_names = [class_des for i, class_des in classes_all.values.tolist()]

    actionClassifier = ActionClassifier(action_names)
    engagementClassifier = EngagementClassifier(action_names, usedLLM='llama', contextAware=False)

    vid = cv2.VideoCapture(v_path)
    if not vid.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {v_path}. "
            "Set v_path to an existing video in engageConfig.py."
        )

    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise RuntimeError(f"Could not read FPS from video: {v_path}")
    print(f"Processing {v_path} ({fps:.2f} FPS)")

    frm_count = 0
    sts_trackers_data = {}
    sts_predActionsSeq = {}
    sts_engagement_levels_dict = {}
    frms = []
    displayFrms = []
    pred_action =""
    predictions = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        display_frame = frame.copy()
        frm_count += 1
        frms.append(frame.copy())
        # ===recognize actions every 3 seconds ===#
        if frm_count >= clip_len and frm_count % clip_len == 0:
            print(f"=================****************************==============")

            inds = actionClassifier.uniformSampling(len(frms))
            sampledSubFrms = [frms[i] for i in inds]
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

                if st_id in sts_predActionsSeq:
                    sts_predActionsSeq[st_id].append(pred_action)
                else:
                    sts_predActionsSeq[st_id] = [pred_action]

            except Exception as e:
                print(f"Error: {e}")
            frms.clear()

        # ===engagement classification===#
        if frm_count % 1800 == 0:
            print(f"////////****************************/////////")
            # ====filter sts_predActionsSeq===#
            try:
                sts_engagement_levels_dict = engagementClassifier.classify_action_sequence(sts_predActionsSeq)
            except Exception as e:
                print(f"Error: {e}")
            sts_predActionsSeq.clear()


        display_frame = cv2.putText(display_frame, f"Pred Act: {pred_action}", (5, height-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 30), 3, )
        if st_id in sts_engagement_levels_dict:
            display_frame = cv2.putText(display_frame, f"{sts_engagement_levels_dict[st_id]}",
                                        (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250),3,)

        if display_enabled:
            cv2.imshow('Student Engagement Measurement Demo', display_frame)
        # cv2.namedWindow('Frms', cv2.WINDOW_NORMAL)
        # cv2.imshow('Frms', frms[-1])

        if display_enabled and cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release resources
    vid.release()
    with open(predictions_file, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=predictions[0].keys()
                                if predictions else [
                                    "student_id", "start_frame", "end_frame",
                                    "start_time", "end_time", "predicted_label"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Saved {len(predictions)} predictions to {predictions_file}")
    if display_enabled:
        cv2.destroyAllWindows()
