import re

# root_path = "/mnt/data_lrc"
# json_path = "/home/sicheng/Desktop/mycode/EPIC-KITCHENS/debug"

root_path = "/mnt/data_3/home_aiglasses"
json_path = "/home/sicheng/Desktop/mycode/EPIC-KITCHENS/debug"

EGO4D = {
    "annotation_path": json_path + "/combined_output.json",
    "data_path": root_path + "/HD-EPIC/Videos/",
}

YOUCOOK2 = {
    # "annotation_path": "/home/kyle/DOING_PROJECTS/Champion/reformulate/annotations/generated_youcookii_qa.json",
    "annotation_path": "/mnt/data_3/home_aiglasses/FINAL_SFT/Youcook2/annotations/generated_youcookii_qav2.json",
    "data_path": "/mnt/data_3/home_aiglasses/YouCookII/",
}

EPICKITCHEN = {
    "annotation_path": "/mnt/data_3/home_aiglasses/FINAL_SFT/Epic-Kitchen/annotations/generated_epickitchen_qa_h200.json",
    "data_path": "/mnt/data_3/home_aiglasses/FINAL_SFT/Epic-Kitchen/",
}


EGOPROCEL = {
    "annotation_path": "/mnt/data_3/home_aiglasses/FINAL_SFT/EgoProceL/annotation/epL_qa_fullv2.json",        # epL_qa, epL_qa_fullv2
    "data_path": "/mnt/data_3/home_aiglasses/dataset/CMU-MMAC/videos/",
}

VISOR = {
    "annotation_path": "/mnt/data_3/home_aiglasses/FINAL_SFT/VSOR/annotations/3d_fixture_vsor.json",
    "data_path": "/mnt/data_3/home_aiglasses/FINAL_SFT/VSOR/images/",
}

data_dict = {
    "youcook2": YOUCOOK2,
    "epickitchen": EPICKITCHEN,
    "egoprocel": EGOPROCEL,
    "ego4d": EGO4D,
    "visor": VISOR,
}




def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["youcook2", "epickitchen", "egoprocel", "ego4d", "visor"]      # 0/4735
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
