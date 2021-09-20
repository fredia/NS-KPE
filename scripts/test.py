import os
import torch
import logging
import argparse
import traceback
from tqdm import tqdm

from scripts import config, utils
from scripts.utils import pred_arranger, pred_saver
from scripts.model_chunk import KeyphraseSpanExtraction
from generator.Chunk2Phrase import chunk2phrase
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from transformers import RobertaTokenizer
from dataloader.bert2chunk_dataloader import batchify_bert2chunk_features_for_test, \
    batchify_bert2chunk_features_for_train
from dataloader.loader_utils import build_dataset

torch.backends.cudnn.benchmark = True

logger = logging.getLogger()
Decode_Candidate_Number = {'openkp': 5, 'kp20k': 50, 'inspec': 50, 'nus': 50, 'krapivin': 50, 'semeval': 50}


# Bert2Chunk
def bert2chunk_decoder(args, data_loader, dataset, model, test_input_refactor,
                       pred_arranger, mode, stem_flag=False):
    logging.info('Start Generating Keyphrases for %s ... \n' % mode)
    test_time = utils.Timer()
    if args.dataset_class == "kp20k": stem_flag = True

    tot_examples = 0
    tot_predictions = []
    for step, batch in enumerate(tqdm(data_loader)):
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)
        try:
            logit_lists = model.test_bert2chunk(inputs, lengths, args.max_phrase_words)
        except:
            logging.error(str(traceback.format_exc()))
            continue

        # decode logits to phrase per batch
        params = {'examples': dataset.examples,
                  'logit_lists': logit_lists,
                  'indices': indices,
                  'max_phrase_words': args.max_phrase_words,
                  'return_num': Decode_Candidate_Number[args.dataset_class],
                  'stem_flag': stem_flag}

        batch_predictions = chunk2phrase(**params)
        tot_predictions.extend(batch_predictions)

    candidate = pred_arranger(tot_predictions)
    return candidate


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # setting args
    parser = argparse.ArgumentParser('BertKPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_default_args(parser)

    args = parser.parse_args()
    config.init_args_config(args)
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
    # init tokenizer & Converter 
    logger.info("start setting tokenizer, dataset and dataloader (local_rank = {})... ".format(args.local_rank))
    tokenizer = RobertaTokenizer[args.pretrain_model_type].from_pretrained(args.cache_dir)

    # -------------------------------------------------------------------------------------------
    # Select dataloader
    batchify_features_for_train, batchify_features_for_test = batchify_bert2chunk_features_for_train, batchify_bert2chunk_features_for_test

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

    # -------------------------------------------------------------------------------------------
    # Preprare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    logger.info(" ************************** Initilize Model ************************** ")
    try:
        model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(args.eval_checkpoint, args)
        model.set_device()
    except ValueError:
        print("Could't Load Pretrain Model %s" % args.eval_checkpoint)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.n_gpu > 1:
        model.parallelize()

    if args.local_rank != -1:
        model.distribute()

    # -------------------------------------------------------------------------------------------
    # Method Select
    # -------------------------------------------------------------------------------------------
    candidate_decoder = bert2chunk_decoder
    evaluate_script, main_metric_name = utils.select_eval_script(args.dataset_class)
    _, test_input_refactor = utils.select_input_refactor(args.model_class)

    # ------------------------------------------------------------------------------------------
    # evaluate checkpoints and generate
    # -------------------------------------------------------------------------------------------
    # dev generator
    dev_candidate = candidate_decoder(args, dev_data_loader, dev_dataset, model, test_input_refactor, pred_arranger,
                                      'dev')
    dev_stats = {'epoch': checkpoint_epoch, main_metric_name: 0}
    dev_stats = evaluate_script(args, dev_candidate, dev_stats, mode='dev', metric_name=main_metric_name)

    # log
    test_names = ['dev']
    test_candidates = [dev_candidate]

    # eval generator
    if args.dataset_class == 'kp20k':
        eval_candidate = candidate_decoder(args, eval_data_loader, eval_dataset, model, test_input_refactor,
                                           pred_arranger, 'eval')

        eval_stats = {'epoch': checkpoint_epoch, main_metric_name: 0}
        eval_stats = evaluate_script(args, eval_candidate, eval_stats, mode='eval', metric_name=main_metric_name)

        # log
        test_names.append("eval")
        test_candidates.append(eval_candidate)

    # ------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # Save : candidate
    for name, candidate in zip(test_names, test_candidates):
        save_filename = os.path.join(args.pred_folder,
                                     '{}.{}_{}.{}.epoch_{}.checkpoint'.format(args.model_class, args.dataset_class,
                                                                              name,
                                                                              args.pretrain_model_type.split('-')[0],
                                                                              checkpoint_epoch))
        pred_saver(args, candidate, save_filename)
