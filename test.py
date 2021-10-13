import matplotlib.pyplot as plt
import cv2
import sys
import torch
from tqdm import trange
from adaptis.inference.adaptis_sampling import get_panoptic_segmentation
from adaptis.inference.prediction_model import AdaptISPrediction
from adaptis.data.toy import ToyDataset
from adaptis.model.toy.models import get_unet_model
from adaptis.coco.panoptic_metric import PQStat, pq_compute, print_pq_stat
from adaptis.utils.vis import visualize_instances, visualize_proposals

#device = torch.device('cuda:0')

# dataset_path = '/home/guest01/projects/adaptis/toyV2/'
# dataset = ToyDataset(dataset_path, split='test', with_segmentation=True)

# model = get_unet_model(norm_layer=torch.nn.BatchNorm2d, with_proposals=True)
# pmodel = AdaptISPrediction(model, dataset, device)

# weights_path = '/home/guest01/projects/adaptis/experiments/toy_v2/001/checkpoints/last_checkpoint.params'
# pmodel.load_parameters(weights_path)

def test_model(pmodel, dataset,
               sampling_algorithm, sampling_params,
               use_flip=False, cut_radius=-1):
    # 全景分割的指标
    pq_stat = PQStat()
    # 获取数据集的stuff和thing的所有标签
    categories = dataset._generate_coco_categories()
    categories = {x['id']: x for x in categories}

    for indx in trange(len(dataset)):
        sample = dataset.get_sample(indx)
        # 获取到图片信息以及具体内部的标注信息和分割信息等，格式如下：
        # 'image': None, 'semantic_segmentation': semantic_segmentation,
        # 'instances_mask': instances_mask, 'instances_info': instances_info,
        # 'masks': masks, 'proposals_info': proposals_info
        pred = get_panoptic_segmentation(pmodel, sample['image'],
                                         sampling_algorithm=sampling_algorithm,
                                         use_flip=use_flip, cut_radius=cut_radius, **sampling_params)
        # 转换为coco格式的数据
        coco_sample = dataset.convert_to_coco_format(sample)
        pred = dataset.convert_to_coco_format(pred)
        pq_stat = pq_compute(pq_stat, pred, coco_sample, categories)
    
    print_pq_stat(pq_stat, categories)

def visulization():
    # Results visualization
    proposals_sampling_params = {
        'thresh1': 0.5,
        'thresh2': 0.5,
        'ithresh': 0.3,
        'fl_prob': 0.10,
        'fl_eps': 0.003,
        'fl_blur': 2,
        'max_iters': 100
    }
    # 15 , 25, 42
    vis_samples = [42, 15, 25]
    for row_indx, sample_indx in enumerate(vis_samples):
        # 需要注意的是，如果进行测试的话，需要将图片的大小变成128*128的大小
        crop_size = (128, 128)
        image = cv2.imread("/home/guest01/projects/adaptis/toyV2/chromos/002.jpg")
        image = cv2.resize(image, crop_size, interpolation = cv2.INTER_CUBIC)
        print("this is image shape", image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = dataset.get_sample(sample_indx)
        pred = get_panoptic_segmentation(pmodel, image,
                                     sampling_algorithm='proposals',
                                     use_flip=True, **proposals_sampling_params)
        # image_ = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite("image.png", image)
        cv2.imwrite("image_mask.png", visualize_instances(pred['instances_mask']))
        cv2.imwrite("image_prop.png", visualize_proposals(pred['proposals_info']))

# 遍历图片的每一个像素
def access_pixels(img):
    """遍历图像每个像素的每个通道"""
    print(img.shape)              #打印图像的高，宽，通道数（返回一个3元素的tuple）
    height = img.shape[0]        #将tuple中的元素取出，赋值给height，width，channels
    width = img.shape[1]
    channels = img.shape[2]
    print("height:%s,width:%s,channels:%s" % (height,width,channels))
    print(img.size)              #打印图像数组内总的元素数目（总数=高X宽X通道数）
    cnt = 0
    for row in range(height):    #遍历每一行
        for col in range(width): #遍历每一列
            for channel in range(channels):    #遍历每个通道（三个通道分别是BGR）
                if img[row][col][channel] != 0:
                    cnt += 1
                    img[row][col][channel] = 150
                #通过数组索引访问该元素，并作出处理
    # cv2.imshow("processed img",img) #将处理后的图像显示出来
    cv2.imwrite("./sample1.png", img)
    print("the non zero pixels nums is： ", cnt)

# visulization()
 
# proposals_sampling_params = {
#     'thresh1': 0.4,
#     'thresh2': 0.5,
#     'ithresh': 0.3,
#     'fl_prob': 0.10,
#     'fl_eps': 0.003,
#     'fl_blur': 2,
#     'max_iters': 100
# }
# test_model(pmodel, dataset,
#            sampling_algorithm='proposals',
#            sampling_params=proposals_sampling_params,
#            use_flip=False)

image_train_sample1 = cv2.imread("/home/guest01/projects/adaptis/toyV2/train/00000_013_im.png")
access_pixels(image_train_sample1)

image_train_sample2 = cv2.imread("/home/guest01/projects/adaptis/toyV2/train/00000_013_rgb.png")
print(image_train_sample2)
