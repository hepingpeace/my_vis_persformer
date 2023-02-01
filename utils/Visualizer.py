import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.utils import *
class Visualizer:
    def __init__(self, args):
        self.save_path = args.save_path
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline
        self.vgg_mean = args.vgg_mean
        self.vgg_std = args.vgg_std
        self.ipm_w = args.ipm_w
        self.ipm_h = args.ipm_h
        self.num_y_steps = args.num_y_steps
        self.h_net = args.resize_h
        self.w_net = args.resize_w

        self.dataset_name = args.dataset_name
        self.use_default_anchor = args.use_default_anchor

        self.num_category = args.num_category
        self.category_dict = {0: 'invalid',
                              1: 'white-dash',
                              2: 'white-solid',
                              3: 'double-white-dash',
                              4: 'double-white-solid',
                              5: 'white-ldash-rsolid',
                              6: 'white-lsolid-rdash',
                              7: 'yellow-dash',
                              8: 'yellow-solid',
                              9: 'double-yellow-dash',
                              10: 'double-yellow-solid',
                              11: 'yellow-ldash-rsolid',
                              12: 'yellow-lsolid-rdash',
                              13: 'fishbone',
                              14: 'others',
                              20: 'roadedge'}

        if args.no_3d:
            # self.anchor_dim = args.num_y_steps + 1
            self.anchor_dim = args.num_y_steps + self.num_category
        else:
            if 'no_visibility' in args.mod:
                # self.anchor_dim = 2 * args.num_y_steps + 1
                self.anchor_dim = 2 * args.num_y_steps + self.num_category
            else:
                # self.anchor_dim = 3 * args.num_y_steps + 1
                self.anchor_dim = 3 * args.num_y_steps + self.num_category

        x_min = args.top_view_region[0, 0]
        x_max = args.top_view_region[1, 0]
        if self.use_default_anchor:
            self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w / 8), endpoint=True)
        else:
            self.anchor_x_steps = args.anchor_grid_x
        self.anchor_y_steps = args.anchor_y_steps

        # transformation from ipm to ground region
        H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                          [self.ipm_w-1, 0],
                                                          [0, self.ipm_h-1],
                                                          [self.ipm_w-1, self.ipm_h-1]]),
                                              np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(H_ipm2g)

        # probability threshold for choosing visualize lanes
        self.prob_th = args.prob_th
    
    def draw_on_img_category(self, img, pred_anchors, gt_anchors, P_g2im, draw_type='laneline'):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:
        """
        fig = plt.figure(dpi=120, figsize=(4,3))
        plt.imshow(img)
        plot_lines = {}
        plot_lines["pred"] = []
        plot_lines["gt"] = []
        lane_anchor = pred_anchors
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            line_pred = {}
            lane_cate = np.argmax(lane_anchor[j, self.anchor_dim-self.num_category:self.anchor_dim])
            if draw_type == 'laneline' and lane_cate != 0:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    if not self.use_default_anchor:
                        anchor_x_2d, _ = homographic_transformation(P_g2im, self.anchor_x_steps[j], self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                visibility = lane_anchor[j, 2 * self.num_y_steps:3 * self.num_y_steps]
                if not self.use_default_anchor:
                    anchor_x_2d = anchor_x_2d.astype(np.int)
                
                x_2d = [x for i, x in enumerate(x_2d) if visibility[i] > self.prob_th]
                y_2d = [y for i, y in enumerate(y_2d) if visibility[i] > self.prob_th]

                line_pred["x_2d"] = x_2d
                line_pred["y_2d"] = y_2d
                line_pred["lane_cate"] = int(lane_cate)
                plot_lines["pred"].append(line_pred)

                if lane_cate == 1: # white dash
                    plt.plot(x_2d, y_2d, 'mediumpurple', lw=4, alpha=0.6)
                    plt.plot(x_2d, y_2d, 'white', linestyle=(0,(10,10)), lw=2, alpha=0.5)
                elif lane_cate == 2: # white solid
                    plt.plot(x_2d, y_2d, 'mediumturquoise', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 3: # double-white-dash
                    plt.plot(x_2d, y_2d, 'mediumorchid', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                elif lane_cate == 4: # double-white-solid
                    plt.plot(x_2d, y_2d, 'lightskyblue', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 5: # white-ldash-rsolid
                    plt.plot(x_2d, y_2d, 'hotpink', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 6: # white-lsolid-rdash
                    plt.plot(x_2d, y_2d, 'cornflowerblue', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=0.75, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=0.5, alpha=0.5)
                elif lane_cate == 7: # yellow-dash
                    plt.plot(x_2d, y_2d, 'yellowgreen', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', linestyle=(0,(20,10)), lw=2, alpha=0.5)
                elif lane_cate == 8: # yellow-solid
                    plt.plot(x_2d, y_2d, 'dodgerblue', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 9: # double-yellow-dash
                    plt.plot(x_2d, y_2d, 'salmon', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                elif lane_cate == 10: # double-yellow-solid
                    plt.plot(x_2d, y_2d, 'lightcoral', lw=4, alpha=0.6)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 11: # yellow-ldash-rsolid
                    plt.plot(x_2d, y_2d, 'coral', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', lw=1, alpha=0.5)
                elif lane_cate == 12: # yellow-lsolid-rdash
                    plt.plot(x_2d, y_2d, 'lightseagreen', lw=4, alpha=0.4)
                    plt.plot(np.array(x_2d) - 3, y_2d, 'white', lw=1, alpha=0.5)
                    plt.plot(np.array(x_2d) + 3, y_2d, 'white', linestyle=(0,(20,10)), lw=1, alpha=0.5)
                elif lane_cate == 13: # fishbone
                    plt.plot(x_2d, y_2d, 'royalblue', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 14: # others
                    plt.plot(x_2d, y_2d, 'forestgreen', lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                elif lane_cate == 20 or lane_cate == 21: # road
                    plt.plot(x_2d, y_2d, 'gold', lw=4, alpha=0.3)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)
                else:
                    plt.plot(x_2d, y_2d, lw=4, alpha=0.4)
                    plt.plot(x_2d, y_2d, 'white', lw=2, alpha=0.5)