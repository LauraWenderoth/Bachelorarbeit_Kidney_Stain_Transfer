import pandas as pd
import glob
import shutil
def extract_names(paths):
    names = []
    for path in paths:
        name = path.split("/")[-1]
        aname = name.split("-")
        cname = aname[1].split("_")[0]
        bname = aname[0]+"-"+cname
        names.append(bname)
    return names

def get_list_of_names(root_path):
    image_paths = glob.glob(root_path + "*", recursive=True)
    names = extract_names(image_paths)
    return names

if __name__ == '__main__':
    IMAGE_ROOT_DIR_PAS = "/home/laurawenderoth/Documents/kidney_microscopy/data/all/PAS/"
    IMAGE_ROOT_DIR_IF = "/home/laurawenderoth/Documents/kidney_microscopy/data/all/IF/"
    BAD_REGISTERD_ROOT_DIR = "/home/laurawenderoth/Documents/kidney_microscopy/data/check registration controls/bad registration/"
    SEMI_BAD_REGISTERD_ROOT_DIR ="/home/laurawenderoth/Documents/kidney_microscopy/data/check registration controls/semi registration/"
    ANNOTATION_ROOT_DIR ="/home/laurawenderoth/Documents/kidney_microscopy/data/control annotations 25-04-2022/"

    df_images = pd.DataFrame(columns=["image_name", "IF", "Registration", "Annotation"])


    PAS_names = get_list_of_names(IMAGE_ROOT_DIR_PAS)
    IF_names = get_list_of_names(IMAGE_ROOT_DIR_IF)
    bad_registered_names = get_list_of_names(BAD_REGISTERD_ROOT_DIR)
    semi_bad_registered_names = get_list_of_names(SEMI_BAD_REGISTERD_ROOT_DIR)
    annotation_names = get_list_of_names(ANNOTATION_ROOT_DIR)

    df_images["image_name"] = PAS_names

    for i, j in df_images.iterrows():
        name = j["image_name"]
        if name in annotation_names:
            df_images.loc[(i, 'Annotation')] = 1
        else:
            df_images.loc[ (i,'Annotation')] = 0
        if name in semi_bad_registered_names:
            df_images.loc[(i, 'Registration')] = "semi"
        elif name in bad_registered_names:
            df_images.loc[(i, 'Registration')] = "bad"
        else:
            df_images.loc[(i, 'Registration')] = "fine"
        if name in IF_names:
            df_images.loc[(i, 'IF')] = 1
        else:
            df_images.loc[ (i,'IF')] = 0

    #save images
    IF_DATA_WITH_ANNOTATION = "/home/laurawenderoth/Documents/kidney_microscopy/data/data_with_annotation/IF/"
    PAS_DATA_WITH_ANNOTATION = "/home/laurawenderoth/Documents/kidney_microscopy/data/data_with_annotation/PAS/"
    image_paths_PAS = glob.glob(IMAGE_ROOT_DIR_PAS + "*", recursive=True)
    image_paths_IF = glob.glob(IMAGE_ROOT_DIR_IF + "*", recursive=True)
    for i, j in df_images.iterrows():
        if df_images.loc[(i, 'Annotation')] == 1 and df_images.loc[(i, 'Registration')] == "fine":
            PAS_path = ""
            IF_path = ""
            for path in image_paths_PAS:
                if df_images.loc[(i, 'image_name')] in path:
                    PAS_path = path
                    break
            for path in image_paths_IF:
                if df_images.loc[(i, 'image_name')] in path:
                    IF_path = path
                    break
            shutil.copy(PAS_path, PAS_DATA_WITH_ANNOTATION)
            shutil.copy(IF_path, IF_DATA_WITH_ANNOTATION)

    df_images.to_excel("/home/laurawenderoth/Documents/Bachelorarbeit/overview_images.xlsx")

