import os
import torch
import logging
import argparse
import traceback
from tqdm import tqdm

from scripts import config, utils
from scripts.utils import pred_arranger, pred_saver
from scripts.model_chunk import KeyphraseSpanExtraction
from scripts.test import bert2chunk_decoder

torch.backends.cudnn.benchmark = True
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaTokenizer
from dataloader.bert2chunk_dataloader import batchify_bert2chunk_features_for_test, \
    batchify_bert2chunk_features_for_train
from dataloader.loader_utils import build_dataset
from tensorboardX import SummaryWriter

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------------------------
def train(args, data_loader, model, train_input_refactor, stats, writer):
    logger.info("start training %s on %s (%d epoch) || local_rank = %d..." %
                (args.model_class, args.dataset_class, stats['epoch'], args.local_rank))

    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    epoch_loss = 0
    epoch_step = 0

    epoch_iterator = tqdm(data_loader, desc="Train_Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        inputs, indices = train_input_refactor(batch, model.args.device)
        try:
            # logger.info(torch.sum((inputs['chunk_labels'] == 1).long(), dim=1))
            loss = model.update(step, inputs)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        train_loss.update(loss)
        epoch_loss += loss
        epoch_step += 1

        if args.local_rank in [-1, 0] and step % args.display_iter == 0:
            if args.use_viso:
                writer.add_scalar('train/loss', train_loss.avg, model.updates)
                writer.add_scalar('train/lr', model.scheduler.get_lr()[0], model.updates)

            logging.info('Local Rank = %d | train: Epoch = %d | iter = %d/%d | ' %
                         (args.local_rank, stats['epoch'], step, len(train_data_loader)) +
                         'loss = %.4f | lr = %f | %d updates | elapsed time = %.2f (s) \n' %
                         (train_loss.avg, model.scheduler.get_lr()[0], model.updates, stats['timer'].time()))
            train_loss.reset()

    logging.info('Local Rank = %d | Epoch Mean Loss = %.8f ( Epoch = %d ) | Time for epoch = %.2f (s) \n' %
                 (args.local_rank, (epoch_loss / epoch_step), stats['epoch'], epoch_time.time()))


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # setting args
    parser = argparse.ArgumentParser('BertKPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_default_args(parser)
    args = parser.parse_args()
    config.init_args_config(args)
    preprocess_folder = args.preprocess_folder
    args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
    # -------------------------------------------------------------------------------------------
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # -------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # -------------------------------------------------------------------------------------------
    utils.set_seed(args)
    # Make sure only the first process in distributed training will download model & vocab

    # -------------------------------------------------------------------------------------------
    # init tokenizer & Converter 
    logger.info("start setting tokenizer, dataset and dataloader (local_rank = {})... ".format(args.local_rank))
    tokenizer = RobertaTokenizer.from_pretrained(args.cache_dir)

    # -------------------------------------------------------------------------------------------
    # Select dataloader
    batchify_features_for_train, batchify_features_for_test = batchify_bert2chunk_features_for_train, batchify_bert2chunk_features_for_test

    # -------------------------------------------------------------------------------------------
    # build train dataloader
    train_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'train'})
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = torch.utils.data.sampler.RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify_features_for_train,
        pin_memory=args.cuda,
    )
    logger.info("Successfully Preprocess Training Features !")

    # -------------------------------------------------------------------------------------------
    # build dev dataloader 
    dev_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'dev'})
    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify_features_for_test,
        pin_memory=args.cuda,
    )
    logger.info("Successfully Preprocess Dev Features !")

    # -------------------------------------------------------------------------------------------
    # build eval dataloader 
    if args.dataset_class == 'kp20k':
        eval_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'eval'})
        eval_sampler = torch.utils.data.sampler.SequentialSampler(eval_dataset)
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.test_batch_size,
            sampler=eval_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )

        args.dataset_class = 'inspec'
        args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
        inspec_eval_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'eval'})
        inspec_eval_sampler = torch.utils.data.sampler.SequentialSampler(inspec_eval_dataset)
        inspec_eval_data_loader = torch.utils.data.DataLoader(
            inspec_eval_dataset,
            batch_size=args.test_batch_size,
            sampler=inspec_eval_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )

        args.dataset_class = 'nus'
        args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
        nus_eval_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'eval'})
        nus_eval_sampler = torch.utils.data.sampler.SequentialSampler(nus_eval_dataset)
        nus_eval_data_loader = torch.utils.data.DataLoader(
            nus_eval_dataset,
            batch_size=args.test_batch_size,
            sampler=nus_eval_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )

        args.dataset_class = 'krapivin'
        args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
        krapivin_eval_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'eval'})
        krapivin_eval_sampler = torch.utils.data.sampler.SequentialSampler(krapivin_eval_dataset)
        krapivin_eval_data_loader = torch.utils.data.DataLoader(
            krapivin_eval_dataset,
            batch_size=args.test_batch_size,
            sampler=krapivin_eval_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )

        args.dataset_class = 'semeval'
        args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
        semeval_eval_dataset = build_dataset(**{'args': args, 'tokenizer': tokenizer, 'mode': 'eval'})
        semeval_eval_sampler = torch.utils.data.sampler.SequentialSampler(semeval_eval_dataset)
        semeval_eval_data_loader = torch.utils.data.DataLoader(
            semeval_eval_dataset,
            batch_size=args.test_batch_size,
            sampler=semeval_eval_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )
        args.dataset_class = 'kp20k'
        args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
        logger.info("Successfully Preprocess Eval Features !")

    # -------------------------------------------------------------------------------------------
    # Set training total steps
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.max_train_epochs = args.max_train_steps // (len(train_data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.max_train_epochs

    # -------------------------------------------------------------------------------------------
    # Preprare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info(" ************************** Initilize Model & Optimizer ************************** ")

    if args.load_checkpoint and os.path.isfile(args.checkpoint_file):
        model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(args.checkpoint_file, args)
    else:
        logger.info('Training model from scratch...')
        model = KeyphraseSpanExtraction(args)

    # initial optimizer
    model.init_optimizer(num_total_steps=t_total)

    # -------------------------------------------------------------------------------------------
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # -------------------------------------------------------------------------------------------

    # set model device
    model.set_device()

    if args.n_gpu > 1:
        model.parallelize()

    if args.local_rank != -1:
        model.distribute()

    if args.local_rank in [-1, 0] and args.use_viso:
        tb_writer = SummaryWriter(args.viso_folder)
    else:
        tb_writer = None

    logger.info("Training/evaluation parameters %s", args)
    logger.info(" ************************** Running training ************************** ")
    logger.info("  Num Train examples = %d", len(train_dataset))
    logger.info("  Num Train Epochs = %d", args.max_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info(" *********************************************************************** ")

    # -------------------------------------------------------------------------------------------
    # Method Select
    candidate_decoder = bert2chunk_decoder
    evaluate_script, main_metric_name = utils.select_eval_script(args.dataset_class)
    train_input_refactor, test_input_refactor = utils.select_input_refactor(args.model_class)

    # -------------------------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------------------------
    model.zero_grad()
    stats = {'timer': utils.Timer(), 'epoch': 0, main_metric_name: 0}
    best_epoch_dict5 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    best_epoch_dict10 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    inspec_best_epoch_dict5 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    inspec_best_epoch_dict10 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    nus_best_epoch_dict5 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    nus_best_epoch_dict10 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    krapivin_best_epoch_dict5 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    krapivin_best_epoch_dict10 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    semeval_best_epoch_dict5 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    semeval_best_epoch_dict10 = {
        'epoch': 0,

        'f': 0, 'p': 0, 'r': 0

    }
    for epoch in range(1, (args.max_train_epochs + 1)):
        stats['epoch'] = epoch

        # train 
        train(args, train_data_loader, model, train_input_refactor, stats, tb_writer)

        # previous metric score
        prev_metric_score = stats[main_metric_name]

        # decode candidate phrases
        dev_candidate = candidate_decoder(args, dev_data_loader, dev_dataset, model, test_input_refactor, pred_arranger,
                                          'dev')
        stats = evaluate_script(args, dev_candidate, stats, mode='dev', metric_name=main_metric_name)

        if args.dataset_class == "openkp":
            if args.use_viso:
                tb_writer.add_scalar('openkp/f3', eval_stats['f3'], epoch)
                tb_writer.add_scalar('openkp/f5', eval_stats['f5'], epoch)

        # new metric score
        new_metric_score = stats[main_metric_name]

        # save checkpoint : when new metric score > previous metric score
        if args.save_checkpoint and (new_metric_score > prev_metric_score) and (
                args.local_rank == -1 or torch.distributed.get_rank() == 0):
            checkpoint_name = '{}.{}.{}.epoch_{}.checkpoint'.format(args.model_class, args.dataset_class,
                                                                    args.pretrain_model_type.split('-')[0], epoch)
            model.save_checkpoint(os.path.join(args.checkpoint_folder, checkpoint_name), stats['epoch'])

        # eval evaluation
        if args.dataset_class == 'kp20k':
            eval_candidate = candidate_decoder(args, eval_data_loader, eval_dataset, model, test_input_refactor,
                                               pred_arranger, 'eval')
            eval_stats = {'epoch': epoch, main_metric_name: 0}
            eval_stats = evaluate_script(args, eval_candidate, eval_stats, mode='eval', metric_name=main_metric_name)
            if args.use_viso:
                tb_writer.add_scalar('kp20k/f5', eval_stats['f5'], epoch)
                tb_writer.add_scalar('kp20k/f10', eval_stats['f10'], epoch)
            if eval_stats['f5'] > best_epoch_dict5['f']:
                best_epoch_dict5['f'] = eval_stats['f5']
                best_epoch_dict5['p'] = eval_stats['p5']
                best_epoch_dict5['r'] = eval_stats['r5']
            if eval_stats['f10'] > best_epoch_dict10['f']:
                best_epoch_dict10['f'] = eval_stats['f10']
                best_epoch_dict10['p'] = eval_stats['p10']
                best_epoch_dict10['r'] = eval_stats['r10']

            logging.info('@5')
            logging.info(best_epoch_dict5)
            logging.info('@10')
            logging.info(best_epoch_dict10)

            logging.info('insepc')
            args.dataset_class = 'inspec'
            args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
            inspec_eval_candidate = candidate_decoder(args, inspec_eval_data_loader, inspec_eval_dataset, model,
                                                      test_input_refactor,
                                                      pred_arranger, 'eval')
            eval_stats = {'epoch': epoch, main_metric_name: 0}
            eval_stats = evaluate_script(args, inspec_eval_candidate, eval_stats, mode='eval',
                                         metric_name=main_metric_name)
            if args.use_viso:
                tb_writer.add_scalar('inspec/f5', eval_stats['f5'], epoch)
                tb_writer.add_scalar('inspec/f10', eval_stats['f10'], epoch)
            if eval_stats['f5'] > inspec_best_epoch_dict5['f']:
                inspec_best_epoch_dict5['f'] = eval_stats['f5']
                inspec_best_epoch_dict5['p'] = eval_stats['p5']
                inspec_best_epoch_dict5['r'] = eval_stats['r5']
            if eval_stats['f10'] > inspec_best_epoch_dict10['f']:
                inspec_best_epoch_dict10['f'] = eval_stats['f10']
                inspec_best_epoch_dict10['p'] = eval_stats['p10']
                inspec_best_epoch_dict10['r'] = eval_stats['r10']
            logging.info('@5')
            logging.info(inspec_best_epoch_dict5)
            logging.info('@10')
            logging.info(inspec_best_epoch_dict10)

            logging.info('nus')
            args.dataset_class = 'nus'
            args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
            nus_eval_candidate = candidate_decoder(args, nus_eval_data_loader, nus_eval_dataset, model,
                                                   test_input_refactor,
                                                   pred_arranger, 'eval')
            eval_stats = {'epoch': epoch, main_metric_name: 0}
            eval_stats = evaluate_script(args, nus_eval_candidate, eval_stats, mode='eval',
                                         metric_name=main_metric_name)
            if args.use_viso:
                tb_writer.add_scalar('nus/f5', eval_stats['f5'], epoch)
                tb_writer.add_scalar('nus/f10', eval_stats['f10'], epoch)
            if eval_stats['f5'] > nus_best_epoch_dict5['f']:
                nus_best_epoch_dict5['f'] = eval_stats['f5']
                nus_best_epoch_dict5['p'] = eval_stats['p5']
                nus_best_epoch_dict5['r'] = eval_stats['r5']
            if eval_stats['f10'] > nus_best_epoch_dict10['f']:
                nus_best_epoch_dict10['f'] = eval_stats['f10']
                nus_best_epoch_dict10['p'] = eval_stats['p10']
                nus_best_epoch_dict10['r'] = eval_stats['r10']
            logging.info('@5')
            logging.info(nus_best_epoch_dict5)
            logging.info('@10')
            logging.info(nus_best_epoch_dict10)

            logging.info('krapivin')
            args.dataset_class = 'krapivin'
            args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
            krapivin_eval_candidate = candidate_decoder(args, krapivin_eval_data_loader, krapivin_eval_dataset, model,
                                                        test_input_refactor,
                                                        pred_arranger, 'eval')
            eval_stats = {'epoch': epoch, main_metric_name: 0}
            eval_stats = evaluate_script(args, krapivin_eval_candidate, eval_stats, mode='eval',
                                         metric_name=main_metric_name)
            if args.use_viso:
                tb_writer.add_scalar('krapivin/f5', eval_stats['f5'], epoch)
                tb_writer.add_scalar('krapivin/f10', eval_stats['f10'], epoch)

            if eval_stats['f5'] > krapivin_best_epoch_dict5['f']:
                krapivin_best_epoch_dict5['f'] = eval_stats['f5']
                krapivin_best_epoch_dict5['p'] = eval_stats['p5']
                krapivin_best_epoch_dict5['r'] = eval_stats['r5']
            if eval_stats['f10'] > krapivin_best_epoch_dict10['f']:
                krapivin_best_epoch_dict10['f'] = eval_stats['f10']
                krapivin_best_epoch_dict10['p'] = eval_stats['p10']
                krapivin_best_epoch_dict10['r'] = eval_stats['r10']

            logging.info('@5')
            logging.info(krapivin_best_epoch_dict5)
            logging.info('@10')
            logging.info(krapivin_best_epoch_dict10)

            logging.info('semeval')
            args.dataset_class = 'semeval'
            args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
            semeval_eval_candidate = candidate_decoder(args, semeval_eval_data_loader, semeval_eval_dataset, model,
                                                       test_input_refactor,
                                                       pred_arranger, 'eval')
            eval_stats = {'epoch': epoch, main_metric_name: 0}
            eval_stats = evaluate_script(args, semeval_eval_candidate, eval_stats, mode='eval',
                                         metric_name=main_metric_name)
            if args.use_viso:
                tb_writer.add_scalar('semeval/f5', eval_stats['f5'], epoch)
                tb_writer.add_scalar('semeval/f10', eval_stats['f10'], epoch)
            if eval_stats['f5'] > semeval_best_epoch_dict5['f']:
                semeval_best_epoch_dict5['f'] = eval_stats['f5']
                semeval_best_epoch_dict5['p'] = eval_stats['p5']
                semeval_best_epoch_dict5['r'] = eval_stats['r5']
            if eval_stats['f10'] > semeval_best_epoch_dict10['f']:
                semeval_best_epoch_dict10['f'] = eval_stats['f10']
                semeval_best_epoch_dict10['p'] = eval_stats['p10']
                semeval_best_epoch_dict10['r'] = eval_stats['r10']
            logging.info('@5')
            logging.info(semeval_best_epoch_dict5)
            logging.info('@10')
            logging.info(semeval_best_epoch_dict10)
            args.dataset_class = 'kp20k'
            args.preprocess_folder = os.path.join(preprocess_folder, args.dataset_class)
