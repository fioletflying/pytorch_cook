import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt

GL_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
GL_NUMBBOX = 2
GL_NUMGRID = 7     
STATIC_DEBUG = False  # 调试用  
print(len(GL_CLASSES))


def convert(size, box):
    """
    将标注数据中的左上角，右下角的坐标点的格式转换成 
    中心点 + w,h，并进行归一化
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)


def convert_annotation(anno_dir, image_id, labels_dir):
    in_file = open(os.path.join(anno_dir,  'Annotations/%s' % (image_id)))
    image_id = image_id.split('.')[0]
    out_file = open(os.path.join(labels_dir, "%s.txt" %(image_id)), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # 这里如何是比较难的物体就不执行检测
        if cls not in GL_CLASSES or int(difficult) == 1:
            continue
        
        cls_id = GL_CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                    float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)
        # 将序列中的元素以指定的字符连接生成一个新的字符串: 这里用空格来连接
        # 后面接的是一个字符串的迭代器
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")
        

def make_label_txt(anno_dir, labels_dir):
    file_names = os.listdir(os.path.join(anno_dir, 'Annotations'))
    for file in file_names:
        # print(file)
        convert_annotation(anno_dir, file, labels_dir)


def show_labels_img(img_dir, label_dir, imgname):
    img = cv2.imread(os.path.join(img_dir,imgname + ".jpg"))
    h, w = img.shape[:2]
    print("width:%s, hegiht: %s" %(w, h))
    with open(os.path.join(label_dir, imgname + ".txt"), 'r') as flabel:
        for label in flabel:
            label_info = label.split(' ')
            label, x, y, box_w, box_h = [float(x.strip()) for x in label_info]
            print(GL_CLASSES[int(label)])
            pt1 = (int(x * w - box_w * w / 2), int(y * h - box_h * h / 2))
            pt2 = (int(x * w + box_w * w / 2), int(y * h + box_h * h / 2))
            cv2.putText(img, GL_CLASSES[int(label)], pt1, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255))
            cv2.rectangle(img, pt1, pt2, (0,0,255), 2)
    img = img[...,::-1]
    plt.imshow(img)

def img_augument(img_dir, save_img_dir, labels_dir):
    imgs_list = [x.split('.')[0]+".jpg" for x in os.listdir(labels_dir)]
    for img_name in imgs_list:
        print("process %s"%os.path.join(img_dir, img_name))
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[0:2]
        input_size = 448  # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
        if h > w:
            padw = (h - w) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
        elif w > h:
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = cv2.resize(img, (input_size, input_size))
        cv2.imwrite(os.path.join(save_img_dir, img_name), img)
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(os.path.join(labels_dir,img_name.split('.')[0] + ".txt"), 'r') as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox) % 5 != 0:
            raise ValueError("File:"
                             + os.path.join(labels_dir,img_name.split('.')[0] + ".txt") + "——bbox Extraction Error!")

        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        if padw != 0:
            for i in range(len(bbox) // 5):
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
                if STATIC_DEBUG:
                    cv2.rectangle(img, (int(bbox[1] * input_size - bbox[3] * input_size / 2),
                                        int(bbox[2] * input_size - bbox[4] * input_size / 2)),
                                  (int(bbox[1] * input_size + bbox[3] * input_size / 2),
                                   int(bbox[2] * input_size + bbox[4] * input_size / 2)), (0, 0, 255))
        elif padh != 0:
            for i in range(len(bbox) // 5):
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
                if STATIC_DEBUG:
                    cv2.rectangle(img, (int(bbox[1] * input_size - bbox[3] * input_size / 2),
                                        int(bbox[2] * input_size - bbox[4] * input_size / 2)),
                                  (int(bbox[1] * input_size + bbox[3] * input_size / 2),
                                   int(bbox[2] * input_size + bbox[4] * input_size / 2)), (0, 0, 255))
        # 此处可以写代码验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验
        if STATIC_DEBUG:
            cv2.imshow("bbox-%d"%int(bbox[0]), img)
            cv2.waitKey(0)
        with open(os.path.join(labels_dir, img_name.split('.')[0] + ".txt"), 'w') as f:
            for i in range(len(bbox) // 5):
                bbox = [str(x) for x in bbox[i*5:(i*5+5)]]
                str_context = " ".join(bbox)+'\n'
                f.write(str_context)

def convert_bbox2labels(bbox):
    """
    将bbox的(cls,x,y,w,h)数据转换程训练时方便计算loss的数据形式(7,7,5*B+cls_num)
    """
    # 这里默认是分成7个格子
    gridesize = 1.0/7
    # 初始化成(7,7,5*B+cls_num):个数不一样 和 类别不一样
    labels = np.zeros((7,7,5*GL_NUMBBOX+len(GL_CLASSES)))
    for i in range(len(bbox)//5):
        # 计算在那个一格子里面
        gridx = int(bbox[i * 5 + 1] // gridesize)
        gridy = int(bbox[i * 5 + 2] // gridesize)
        # 每一个目标的相对位置: (bbox中心坐标 - 网格左上角点的坐标)/网格大小
        gridpx = bbox[i * 5 + 1] / gridesize - gridx
        gridpy = bbox[i * 5 + 2] / gridesize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5]  = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i * 5])] = 1
    labels = labels.reshape(1, -1)
    return labels


def create_csv_txt(img_dir, anno_dir, save_root_dir, train_val_ratio=0.9, padding=10, debug=False):
    """
    TODO:
    将img_dir文件夹内的图片按实际需要处理后，存入save_dir
    最终得到图片文件夹及所有图片对应的标注(train.csv/test.csv)和图片列表文件(train.txt, test.txt)
    """
    labels_dir = os.path.join(anno_dir, "labels")
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
        make_label_txt(anno_dir, labels_dir)
        print("labels done.")
    save_img_dir = os.path.join(os.path.join(anno_dir, "voc2012_forYolov1"), "img")
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
        # img_augument(img_dir, save_img_dir, labels_dir)
    imgs_list = os.listdir(save_img_dir)
    n_trainval = len(imgs_list)
    shuffle_id = list(range(n_trainval))
    random.shuffle(shuffle_id)
    n_train = int(n_trainval*train_val_ratio)
    train_id = shuffle_id[:n_train]
    test_id = shuffle_id[n_train:]
    traintxt = open(os.path.join(save_root_dir, "train.txt"), 'w')
    traincsv = np.zeros((n_train, GL_NUMGRID*GL_NUMGRID*(5*GL_NUMBBOX+len(GL_CLASSES))),dtype=np.float32)
    for i,id in enumerate(train_id):
        img_name = imgs_list[id]
        img_path = os.path.join(save_img_dir, img_name)+'\n'
        traintxt.write(img_path)
        with open(os.path.join(labels_dir,"%s.txt"%img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            traincsv[i,:] = convert_bbox2labels(bbox)
    np.savetxt(os.path.join(save_root_dir, "train.csv"), traincsv)
    print("Create %d train data." % (n_train))

    testtxt = open(os.path.join(save_root_dir, "test.txt"), 'w')
    testcsv = np.zeros((n_trainval - n_train, GL_NUMGRID*GL_NUMGRID*(5*GL_NUMBBOX+len(GL_CLASSES))),dtype=np.float32)
    for i,id in enumerate(test_id):
        img_name = imgs_list[id]
        img_path = os.path.join(save_img_dir, img_name)+'\n'
        testtxt.write(img_path)
        with open(os.path.join(labels_dir,"%s.txt"%img_name.split('.')[0]), 'r') as f:
            bbox = [float(x) for x in f.read().split()]
            testcsv[i,:] = convert_bbox2labels(bbox)
    np.savetxt(os.path.join(save_root_dir, "test.csv"), testcsv)
    print("Create %d test data." % (n_trainval-n_train))


if __name__ == '__main__':
    random.seed(0)
    np.set_printoptions(threshold=np.inf)
    STATIC_DATASET_PATH = '/data/data/voc/VOCdevkit/VOC2007'
    img_dir = os.path.join(STATIC_DATASET_PATH, "JPEGImages")
    anno_dirs = [STATIC_DATASET_PATH]
    save_dir = os.path.join(STATIC_DATASET_PATH, "voc2012_forYolov2")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for anno_dir in anno_dirs:
        create_csv_txt(img_dir, anno_dir, save_dir, debug=False)