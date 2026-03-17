MVSEC_HEIGHT = 260
MVSEC_WIDTH = 346

MVSEC_TRAIN = {
    "train/outdoor_day2":12178,
}

MVSEC_TEST = {
    "test/outdoor_day1":5125, "test/outdoor_night1":5111, "test/outdoor_night2":5478, "test/outdoor_night3":5411,
}

MVSEC_ALL_DATA_FOLDERS = list(MVSEC_TRAIN.keys()) + list(MVSEC_TEST.keys())