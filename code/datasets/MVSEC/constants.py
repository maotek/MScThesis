MVSEC_HEIGHT = 260
MVSEC_WIDTH = 346

MVSEC_TRAIN = {
    "train/outdoor_day2":12197,
}

MVSEC_TEST = {
    "test/outdoor_day1":5134, "test/outdoor_night1":5133, "test/outdoor_night2":5497, "test/outdoor_night3":5429,
}

MVSEC_ALL_DATA_FOLDERS = list(MVSEC_TRAIN.keys()) + list(MVSEC_TEST.keys())