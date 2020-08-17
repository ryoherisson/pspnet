"""Dataset process
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

from pathlib import Path

def make_datapath_list(rootpath, train_data='train.txt', test_data='test.txt', extension='.jpg'):
    """
    Create list of image and annotation data path

    Parameters
    ----------
    rootpath : str
        path to the data directory
    train_data : 
        text file with train filename
    test_data : 
        text file with test filename
    extension :
        extension of image
    Returns
    ----------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
    """

    img_dir = Path(rootpath) / 'images'
    annot_dir = Path(rootpath) / 'annotations'

    train_filenames = Path(rootpath) / train_data
    test_filenames = Path(rootpath) / test_data

    # create train img and annot path list
    train_img_list = []
    train_annot_list = []

    for line in open(train_filenames):
        line = line.rstrip('\n')
        img_fname = line + extension
        img_path = img_dir / img_fname
        annot_path = annot_dir / img_fname
        train_img_list.append(str(img_path))
        train_annot_list.append(str(annot_path))

    # create test img and annot path list
    test_img_list = []
    test_annot_list = []

    for line in open(test_filenames):
        line = line.rstrip('\n')
        img_fname = line + extension
        img_path = img_dir / img_fname
        annot_path = annot_dir / img_fname
        test_img_list.append(str(img_path))
        test_annot_list.append(str(annot_path))

    return train_img_list, train_annot_list, test_img_list, test_annot_list



