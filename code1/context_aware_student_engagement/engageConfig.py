import os


openai_api_key = os.getenv("OPENAI_API_KEY", "")
display_enabled = (
    os.getenv("ENGAGEMENT_HEADLESS", "0") != "1"
    and bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
)
#======action recognition config===#
pretrained_model_path  = './pretrained_action_recog_models/ckpt_epoch_30.pth'
config = './pretrained_action_recog_models/16_32_vifi_clip_16_shot_cvpr_n_actions.yaml'
labels_file = "action_labels.csv"
#======video Conf===#
v_path = "./videos/multi2_FPS_Videobolt.net.mp4"
clip_len = 15*3
predictions_file = "predictions_single_student.csv"
multi_predictions_file = "predictions_multi_student.csv"
#=====object detcetor conf===#
yolox_model = ""

ar_period= 15*3
engage_period = 15*120
