### yolov8损失代码解析
#### yolov8版本为`__version__ = '8.0.110'`

* 先验知识
```python
统一注释
a = na = 3wh = h*w = 8400 = 20*20 + 40*40 + 80*80 因为yolov8是free anchor的, 因此, 这个就是num_total_anchor
batchsize: bs = b = 32
lhw = 80*80, mwh=40*40, swh=20*20 

n_max_boxes = max_num_obj =22 类似于: max(batch.len(label)) 
len(label)是batch中每一张图片的label数量, max(batch.len(label))是选择其中的最大值
```
损失函数代码讲解
```python
'''
逻辑：
1. 将预测的三个特征图合并并进行split, 得到pred_scores(bs, 3hw, 2), pred_distri(bs, 3hw, 64)
2. 生成anchor_points[3hw, 2]和stride_tensor[3hw, 1]
3. 生成gt_labels, gt_bboxes, 这两部分其中有0补齐
4. 将pred_distri转化为xyxy, pred_bboxes(bs, 3hw, 4)
5. 使用TAL进行正负样本分配, 得到target_bboxes, target_scores(bboxs, scores都是与pred的bboxs, scores一一对应的), fg_mask是最终的正负样本分配结果
6. 计算cls_loss和boxes_loss
'''
# Criterion class for computing training losses
class Loss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no  # 每一个检测头的输出的[b, n, h, w]中的n: 2(class) + 4(box) * 16(reg_max) = 66
        self.reg_max = m.reg_max  # dfl reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        这个函数的作用是输出一个tensor, 这个tensor由targets和zeros组成
        counts是Batch中每一张图片的labels数量, 选择其中最大的数量生成一个out, 其shape为[Batch, max_labels, 5]的全零tensor
        将targets中对应图片的labels(cls, xyxy)复制到out[..., :5]中, out[..., 0]为cls, out[..., 1:5]为xyxy
        targets中有小于max_labels数量的, 即out[B, len(targets):, 5]全都为0
        Args:
            targets: torch.Size([na, 6])
            batch_size: int
            scale_tensor: tensor([640, 640, 640, 640])
        return:
            out: torch.Size([Batch, max_labels, 5])
        """
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)  # 图片索引出现的次数  torch.Size([32])
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    # 将targets中对应图片的labels(cls, xyxy)复制到out[..., :5]中, out[..., 0]为cls, out[..., 1:5]为xyxy
                    # j是对应的图片, n是该图片的label数量, out[j, n:]全部是0
                    out[j, :n] = targets[matches, 1:]
            # scale_tensor: [640, 640, 640, 640]
            # mul_是将out[..., 1:5]中的值逐元素的乘以scale_tensor
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out


    def bbox_decode(self, anchor_points, pred_dist):
        # 这个bbox_decode函数的目的是从预测的物体边界框坐标分布（pred_dist）和参考点（anchor_points）解码出实际的边界框坐标xyxy。
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # pred_dist在matmul之前, shape为[b, a, 4, 16], self.proj的shape为[16], 最终的pred_dist的shape为[b, a, 4]
            # 如果不理解可以直接使用 a = torch.ones((1, 3, 4, 16)), 与b=torch.rand(16)进行matmul
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    # 假设检测是二分类['flame','smoke']
    # datasets如果使用了moasic技术的话, 那么batch_idx的label数量就对不上这个im_file中的图片上的label数量
    # list((32, ))表示列表, (32, )类似shape
    # stride: 是检测头输出的特征图相对于模型输入图片的缩小倍数
    def __call__(self, preds, batch):
        """
        Args:
            preds: [tensor(Size([32, 66, 80, 80])), tensor(Size([32, 66, 40, 40])), tensor(Size([32, 66, 20, 20]))]
            bacth: {'im_file': list((32, )), 'ori_shape': list((32, )), 'resized_shape': list((32, )), 
                    'img': tensor(Size[32, 3, 640, 640]), 'cls': tensor(Size[211, 1]), 'bboxes': tensor(Size[211, 4]), 'batch_idx': Size[211]}
        """

        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds  # preds是列表, feats=preds
        # feats[0].shape[0]: bs  self.no: 66
        # xi: tensor(Size[bs, 66, w*h]]) h, w对应的三个检测头的特征图的大小分别为20, 40, 80(大目标, 中目标, 小目标)
        # pred_distri: tensor(Size[bs, 64, 8400])  pred_scores: tensor(Size[bs, 2, 8400])
        # pred_distri表示的是边界框的距离, 将来会decode为边界框坐标, pred_scores表示的是边界框的分类分数
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (bs, 3hw, 2)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (bs, 3hw, 64)

        dtype = pred_scores.dtype  # dtype: torch.float16
        batch_size = pred_scores.shape[0]  # batch_size: 32
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w) imgsz: [640, 640]
        
        # anchor_point: torch.Size([8400, 2])
        # anchor_point代表的是三个特征图80*80, 40*40, 20*20的anchor的中心点坐标
        # stride_tensor: torch.Size([8400, 1])
        # 中心点坐标对应的stride
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # targets: shape[na, 6] 这里的na表示的是32张图片mosaic后的总的label数量, 6表示的是(batch_idx, cls, xyxy)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        # targets: torch.Size([bs, n_max_boxes, 5])
        # 这里的22就是32张图片中的最大label数量
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # gt_labels: torch.Size([bs, n_max_boxes, 1]), gt_bboxes: torch.Size([bs, n_max_boxes, 4])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

        # mask_gt: torch.Size([bs, n_max_boxes, 1])
        # 这个是用来判断是否有gt的, 先将xyxy的所有值求和, gt_(0)是大于0的置1, 否则置0
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        # 将pred_distri解码为边界框坐标, pred_bboxes: torch.Size([bs, a, 4])
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (bs, h*w, 4)

        # fg_mask: torch.Size([bs, h*w])
        # 值为True的正样本, 为False的是负样本
        # target_bboxes: torch.Size([bs, h*w, 4]) 与pred_bboxes的shape 一一对应
        # 这个target_bboxes: h*w中还有负样本, 负样本不参与bbox计算, 在后续会进行处理
        # target_scores(shape(bs, h*w, num_class))
        # 负样本的num_class的值全为0
        # 举个例子, bs=1, h*w=2, num_class=2  target_scores: tensor([[0, 0], [0, 1]])(当然这个1可能是个小数)
        # [0, 0]对应的pd0是负样本, [0, 1]对应的pd1是正样本，对应的是gt1, 即假如是二分类{'0':flame, '1':smoke}对应的是'1':smoke
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        # 在分类损失求平均的时候使用
        # target_scores_sum: tensor(734.898, device='cuda:0')
        # 这里求和有小数的原因是在正负样本分配的最后target_scores乘以了一个动态权重
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # bceloss, 正负样本都参加分类损失的计算
        # 计算方式详情可以看https://blog.csdn.net/shilichangtin/article/details/135185583
        # 除以target_scores_sum就是为了求平均
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        # 只有正样本参加bbox损失计算
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
```

* bbox损失
```python
class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        # weight: torch.Size([1700, 1])
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # iou: torch.Size([1700, 1])
        # 这个就是只有正阳本参与box计算
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 这个类似于分类损失求平均, 再次提醒 target_scores中的值可能是小数, 因此weight也是小数
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        # DFL loss论文没看, 现不做解读, 后续补充
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

```
#### 1.为什么使用target_scores_sum
以往的yolov5在BCELoss的使用`BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))`，默认使用的是`reduction='mean'`,这代表求完损失之后是直接求平均的。

在yolov8检测Loss初始化的时候, 使用的代码是`self.bce = nn.BCEWithLogitsLoss(reduction='none')`，这个`reduction='none`的使用方法可以参考[我的torch.nn.BCEWithLogitsLoss用法介绍博客](https://blog.csdn.net/shilichangtin/article/details/135185583?spm=1001.2014.3001.5502)， 因此需要进行平均，这个平均的方法是除以target_scores_sum(当然这个平均的分母中只有正样本进行参与，负样本的全是0，求和相当于直接排除负样本了, 而`self.bce(pred_scores, target_scores.to(dtype))`是正负样本都有值，因此求`sum()`的时候正负样本都参与了，似乎是不公平的？这个还是不是特别理解, yolov5计算的时候使用的是`reduction='mean'`是直接除以总数)

后续的box_loss计算也用上了这个target_scores_sum和target_score，作用也是求平均，与cls_loss类似(box_loss计算时候只有正样本参与计算损失, 因此不存在上面的这个疑惑)

