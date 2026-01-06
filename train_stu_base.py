import datetime
import os
import time
import torch
import torch.utils.data
import wandb
import cv2
import random
import transforms as T
import utils
import numpy as np
import gc
import operator
from functools import reduce
from bert.modeling_bert import BertModel
from lib import segmentation
from loss.loss import Loss
from torch import nn
import torch.distributed as dist
os.environ["WANDB_MODE"]="offline"
from caption_prompt_learner import PromptLearnerForBERT


def seed_everything(seed=2401):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(image_set, transform, args):
    from data.dataset_train_student import ReferDataset_stu
    ds = ReferDataset_stu(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def get_transform(args):
    transforms = [
                  T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)


def criterion(input, target, weight=0.1):
    return Loss(weight=weight)(input, target)


def evaluate(model, data_loader, bert_model, epoch, caption_bank):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    total_its = 0
    acc_ious = 0


    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            if len(data) == 5:
                image, target, sentences, attentions, sample_id = data
            else:
                image, target, sentences, attentions = data
            pixels = cv2.countNonZero(target.data.numpy()[0]) / 230400.
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)


            if bert_model is not None:
                input_embeds = bert_model.module.embeddings.word_embeddings(sentences)
                inputs_embeds, new_attention_mask = caption_bank(sample_id, input_embeds, attentions)
                last_hidden_states = bert_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=new_attention_mask
                )[0]
                bert_embedding = last_hidden_states.permute(0, 2, 1)
                output = model(image, bert_embedding, new_attention_mask.unsqueeze(-1))
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            loss = criterion(output, target)
            total_loss += loss.item()
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    if args.local_rank == 0:
        wandb.log({
            "val mIoU": mIoU,
            "val oiou": cum_I * 100. / cum_U,
            "val Loss": total_loss / total_its})

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, caption_bank):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0


    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_its += 1
        image, target, sentences, attentions, sample_id = data
        image, target, sentences, attentions, sample_id = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True),\
                                               sample_id.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            input_embeds = bert_model.module.embeddings.word_embeddings(sentences)
            inputs_embeds, new_attention_mask = caption_bank(sample_id, input_embeds, attentions)
            last_hidden_states = bert_model(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask
            )[0]
            bert_embedding = last_hidden_states.permute(0, 2, 1)
            output = model(image, bert_embedding, new_attention_mask.unsqueeze(-1))
        else:
            output = model(image, sentences, attentions)

        optimizer.zero_grad()
        loss = criterion(output, target)
        #if hasattr(caption_bank, "module"):
        #    reg = caption_bank.module.prompt_embeddings.norm()
        #else:
        #    reg = caption_bank.prompt_embeddings.norm()
        #loss = loss + 0.0 * reg
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()


        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if bert_model is not None:
            del last_hidden_states, bert_embedding
        del image, target, sentences, attentions, sample_id, output, loss, data

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if args.local_rank == 0:
        wandb.log({
            "Train Loss": train_loss / total_its,})

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    
    dist_backend = 'nccl'
    dist_url = 'env://'
    
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=dist_backend, 
                init_method=dist_url,
                world_size=world_size, 
                rank=rank
            )
            dist.barrier()
        else:
            print("Process group already initialized, skipping initialization.")
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Falling back to single GPU training")
        return False, 0, 1, 0
    
    return True, rank, world_size, local_rank

def main(args):

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        distributed, rank, world_size, local_rank = init_distributed()
        args.distributed = distributed
        args.rank = rank
        args.world_size = world_size
        args.gpu = local_rank
        args.local_rank = local_rank
        
        if args.distributed:
            print(f"Running in distributed mode with {world_size} GPUs, rank: {rank}, local_rank: {local_rank}")
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    except Exception as e:
        print(f"Error in distributed initialization: {e}")
        print("Falling back to single GPU training")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.local_rank = 0
        
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)

    sample_id_list = dataset.ref_ids
    
    caption_bank = PromptLearnerForBERT(sample_id_list, embedding_dim=768, n_prompt=20, position="end").cuda()


    caption_bank = torch.nn.SyncBatchNorm.convert_sync_batchnorm(caption_bank)
    caption_bank = torch.nn.parallel.DistributedDataParallel(caption_bank, device_ids=[args.local_rank])

    dataset_test, _ = get_dataset("train",
                                  get_transform(args=args),
                                  args=args)

    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=5, sampler=test_sampler, num_workers=args.workers)

    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module  

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model
    else:
        bert_model = None
        single_bert_model = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'], strict=False)
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])


    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.module.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    for p in single_model.parameters():
        p.requires_grad = False
    if single_bert_model is not None:
        for p in single_bert_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(caption_bank.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    resume_epoch = -999

    if args.local_rank == 0:
        wandb.watch(model, log="all")


    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model, caption_bank)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, epoch, caption_bank)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        best = (best_oIoU < overallIoU)
        if single_bert_model is not None:
            dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}
        else:
            dict_to_save = {'model': single_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                            'lr_scheduler': lr_scheduler.state_dict()}

        if best:
            print('Better epoch: {}\n'.format(epoch))
            #utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
            #                                                'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
            if args.local_rank == 0:
                embed_path = os.path.join(args.output_dir, f"caption_embeddings_{args.model_id}.pt")
                torch.save({
                    "embedding": caption_bank.module.prompt_embeddings.detach().cpu()
                                if isinstance(caption_bank, torch.nn.parallel.DistributedDataParallel)
                                else caption_bank.prompt_embeddings.detach().cpu(),
                    "sample_id_to_index": caption_bank.module.sample_id_to_index
                                        if isinstance(caption_bank, torch.nn.parallel.DistributedDataParallel)
                                        else caption_bank.sample_id_to_index
                }, embed_path)
                print(f"=> Saved caption embedding to {embed_path}")
        #utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
        #                                                'model_last_{}.pth'.format(args.model_id)))
        if args.local_rank == 0:
            wandb.save('model.h5')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    if args.local_rank == 0:
        wandb.init(project="rmsin_2080")
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
