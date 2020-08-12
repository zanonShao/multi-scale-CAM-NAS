from MyDataloader import mydataset
import numpy as np
import torch
from trapezoid_supernet import trapezoid_supernet
import os
import torch.nn.functional as F
import cv2

features_blobs = []


def hook_feature(module, input, output):  # input是注册层的输入 output是注册层的输出
    global features_blobs
    features_blobs.append(output)


def CAM_example(dataset, model, number, max_scale):
    size_upsample = (224, 224)
    images, lables, raws = dataset.get_cam_examples(number)
    images = torch.stack(images).cuda()

    # extract Linear's weight matrix
    model.eval()
    parameters = list(model.parameters())
    weight = parameters[-2]

    # extract Final 2D Features maps, use hooks
    handles = []
    for i in range(max_scale):
        handles.append(model._modules.get('feature_mix')[i].register_forward_hook(hook_feature))
    logits = model(images)
    for i in range(max_scale):
        handles[i].remove()

    # generate CAM heat-maps (images,2D_Features_maps,weight matrix)
    index = logits.argmax(axis=-1)
    cams = []
    for i in range(number):
        scal_list = []
        for j in range(max_scale):
            feature = F.relu(features_blobs[j][i])
            weight_scal = weight[index[i], 20 * j:20 * (j + 1)].reshape(1, -1)
            cam = torch.matmul(weight_scal, feature.reshape(feature.shape[0], -1))
            cam = cam.reshape(1, 1, feature.shape[-2], feature.shape[-1])
            # print(cam.shape)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
            cam = F.interpolate(cam, size=size_upsample, mode='bilinear', align_corners=True)
            scal_list.append(cam)
        cams.append(np.uint8(sum(scal_list).data.cpu().numpy().squeeze() * 255))
    # print(cams)
    htitch_cam = cams[0]
    htitch_raw = raws[0]
    for i in range(1, number):
        htitch_cam = np.hstack((htitch_cam, cams[i]))
        htitch_raw = np.hstack((htitch_raw, raws[i]))
    # htitch_raw = cv2.cvtColor(htitch_raw, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(htitch_cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    results = heatmap * 0.3 + htitch_raw * 0.5
    return htitch_raw, heatmap, results
    # cv2.imwrite('raw.png', htitch_raw)
    # cv2.imwrite('heatmap.png', heatmap)
    # cv2.imwrite('relut.png', reluts)


def CAM_example_2(dataset, model, number, max_scale):
    size_upsample = (224, 224)
    images, lables, raws = dataset.get_cam_examples(number)
    images = torch.stack(images).cuda()

    # extract Linear's weight matrix
    model.eval()
    parameters = list(model.parameters())
    weight = parameters[-2]

    # extract Final 2D Features maps, use hooks
    handles = []
    for i in range(max_scale):
        handles.append(model._modules.get('feature_mix')[i].register_forward_hook(hook_feature))
    logits = model(images)
    for i in range(max_scale):
        handles[i].remove()

    # generate CAM heat-maps (images,2D_Features_maps,weight matrix)
    index = logits.argmax(axis=-1)
    cams = []
    for i in range(number):
        scal_list = []
        for j in range(max_scale):
            feature = F.relu(features_blobs[j][i])
            weight_scal = weight[index[i], 20 * j:20 * (j + 1)].reshape(1, -1)
            cam = torch.matmul(weight_scal, feature.reshape(feature.shape[0], -1))
            cam = cam.reshape(1, 1, feature.shape[-2], feature.shape[-1])
            # print(cam.shape)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
            cam = F.interpolate(cam, size=size_upsample, mode='bilinear', align_corners=True)
            scal_list.append(cam)
        # cams.append(np.uint8(sum(scal_list).data.cpu().numpy().squeeze() * 255))
        cams.append([np.uint8(scal.data.cpu().numpy().squeeze() * 255) for scal in scal_list])

    # print(cams)
    htitch_cam = np.vstack(cams[0])
    htitch_raw = raws[0]
    for i in range(1, number):
        htitch_cam = np.hstack((htitch_cam, np.vstack(cams[i])))
        htitch_raw = np.hstack((htitch_raw, raws[i]))
    # htitch_raw = cv2.cvtColor(htitch_raw, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(htitch_cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    results = np.uint8(heatmap * 0.3 + np.vstack([htitch_raw for i in range(max_scale)]) * 0.5)
    return htitch_raw, heatmap, results
