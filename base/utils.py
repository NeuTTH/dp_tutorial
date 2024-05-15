from scipy.signal import butter, filtfilt
import numpy as np
import seaborn as sns

state_dict = {
    1: "Wing grooming (left)",
    2: "Wing grooming (right)",
    3: "Hind grooming (left)",
    4: "Hind grooming (bilateral)",
    6: "Hind grooming (right)",
    7: "Locomotion (fastest)",
    9: "Locomotion (slowest)",
    10: "Locomotion (medium)",
    16: "idle",
    18: "Anterior grooming",
}

joint_labels = np.array(
    [
        "Head",
        "Left Eye",
        "Right Eye",
        "Neck",
        "Abdomen",
        "Foreleg R1",
        "Foreleg R2",
        "Foreleg R3",
        "Foreleg R4",
        "Midleg R1",
        "Midleg R2",
        "Midleg R3",
        "Midleg R4",
        "Hindleg R1",
        "Hindleg R2",
        "Hindleg R3",
        "Hindleg R4",
        "Foreleg L1",
        "Foreleg L2",
        "Foreleg L3",
        "Foreleg L4",
        "Midleg L1",
        "Midleg L2",
        "Midleg L3",
        "Midleg L4",
        "Hindleg L1",
        "Hindleg L2",
        "Hindleg L3",
        "Hindleg L4",
        "Left Wing",
        "Right Wing",
    ]
)

c = sns.color_palette("tab20c")
beh_palette = sns.color_palette([c[0], c[1], c[4], c[5], c[6], c[8], c[9], c[10]])


def butter_pass(pass_type, data, cutoff, fs, order, axis=0):
    b, a = butter(order, cutoff / (fs * 0.5), btype=pass_type, analog=False)
    joints_pass = []
    for idx, seg in enumerate(data):
        joints_pass.append(filtfilt(b, a, seg, axis=axis))
    return np.array(joints_pass, dtype=object)
