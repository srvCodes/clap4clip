import torch
import random
import numpy as np
import argparse
import os
import shutil
import pickle
from classifier.zero_shot import ZeroshotCLIP
from classifier.clip_adapter import ClipAdapter
from classifier.continual_clip_variational import ClClipVariational
import dataset.incremental_dataloader as incremental_dataloader
from utils import mkdir_p
from dataset.exemplars_selection import *
from utils.rotation_angle_matrix import RotationAngleMatrix

def parse_option():
    parser = argparse.ArgumentParser('Prompt Learning for CLIP', add_help=False)

    parser.add_argument("--root", type=str, default='/data1/imagenet100',help='root')
    parser.add_argument("--aug",type=str, default='flip', help='root')

    parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
    parser.add_argument("--db_name", type=str, default='cifar100', help='dataset name')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    parser.add_argument("--arch", type=str, default='ViT-B-16', help='arch')
    parser.add_argument("--checkpoint", type=str, default='ckpt/', help='save_checkpoint')
    parser.add_argument("--ckpt_path", type=str, default=None, help='ckpt_path')
    parser.add_argument("--save_path", type=str, default='save/', help='save_path')

    # optimization setting
    parser.add_argument("--lr", type=float, default=1e-3, help='num_runs')#1e-3
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--epochs", type=int, default=5, help='num_runs')
    parser.add_argument("--train_batch", type=int, default=32, help='num_runs')
    parser.add_argument("--test_batch", type=int, default=32, help='num_runs')

    #model setting
    parser.add_argument("--model", type=str, default='coop', help='model')
    parser.add_argument("--n_prompt", type=int, default=32, help='num_runs')
    parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

    #incremental setting
    parser.add_argument("--num_class", type=int, default=100, help='num_class')
    parser.add_argument("--class_per_task", type=int, default=10, help='class per task')
    parser.add_argument("--num_task", type=int, default=10, help='num_task')
    parser.add_argument("--start_sess", type=int, default=0, help='start session')
    parser.add_argument("--sess", type=int, default=0, help='current session')
    parser.add_argument("--memory", type=int, default=1000, help='memory')
    parser.add_argument("--memory-type", type=str, default='fix_total', help='"fix_total", "fix_per_cls"')
    parser.add_argument("--num_test", type=int, default=15, help='num_test_text')
    parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
    parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
    parser.add_argument("--keep", type=bool, default=False, help='keep')

    parser.add_argument("--multi-gpu", action='store_true', default=False, help="use multi-gpus")
    parser.add_argument("--gpus", default=[0], type=lambda x: list(map(int, x.split(','))), help="gpu id(s)")
    parser.add_argument("--default-gpu", default=0, type=int, help="default gpu to use")
    parser.add_argument("--method", type=str, default='no_replay')
    parser.add_argument("--finetuning", action='store_true', default=False, help="Use class-balanced finetuning")
    parser.add_argument("--finetune-epochs", type=int, default=1, help="Use class-balanced finetuning")
    parser.add_argument("--expandable-prompt", action='store_true', default=False)
    parser.add_argument("--variational", action='store_true', default=False)
    
    parser.add_argument("--use-vga", action='store_true', default=False, help="Use VGA module")
    parser.add_argument("--expandable-tokens", action='store_true', default=False)
    parser.add_argument("--expandable-adapter", action='store_true', default=False)
    
    parser.add_argument("--distill", action='store_true', default=False, help="distill with old model")
    parser.add_argument("--lasp", action='store_true', default=False, help="use LASP loss")
    parser.add_argument("--unc-aware-prior", action='store_true', default=False, help="select samples for prior based on uncertainty")

    parser.add_argument("--alpha", type=float, default=10., help="any ratio")
    parser.add_argument("--beta", type=float, default=15., help="any ratio")
    parser.add_argument("--gamma", type=float, default=0.01, help="matching a prior distribution")
    
    parser.add_argument("--top-k", type=int, default=1, help="top-k samples per label to consider for NP prior derivation")
    parser.add_argument("--exemplar-selector", type=str, default='random', help="Exemplar selection technique for rehearsal")
    parser.add_argument("--compute-ram", action='store_true', default=False, help="Compute Rotation Angle Matrix")
    parser.add_argument("--compute-bwt", action='store_true', default=False, help="Compute Backward Transfer")
    
    parser.add_argument("--ortho-loss", default=False, action="store_true", help="orthogonal loss for attri-clip")
    parser.add_argument("--matching-loss", default=False, action="store_true", help="matching loss for attri-clip")
    parser.add_argument("--use-np-prior", action='store_true', default=False, help="Use task specific data driven priors")
    parser.add_argument("--get-interclass-dist", action='store_true', default=False, help="Compute class-sp. means for viz. ")
    parser.add_argument("--distill-distribution", action='store_true', default=False, help="Distillation using recorded task distributions")

    parser.add_argument("--compute-ece", action='store_true', default=False, help="Compute Expected Calibration Error")
    parser.add_argument("--num-run", default=0, type=int, help="number of run decides the class order for cifar100 and seed for imagenet100" )
    parser.add_argument("--get-adapter-distances", action='store_true', default=False, help="average distance between samples of each adapter")

    parser.add_argument("--forward-times", type=int, default=10, help="MC samples")
    parser.add_argument("--forward-times-global", type=int, default=10, help="global MC samples")
    parser.add_argument("--hierarchical", action='store_true', default=False, help="use a global encoder")
    parser.add_argument("--eval-ood-score", action='store_true', default=False, help="evaluate OOD scores")
    parser.add_argument("--use-det-path", action="store_true", default=False, help="use deterministic path")
    parser.add_argument("--context-size", type=float, default=0.67, help="Context size for NP prior")
    parser.add_argument("--frozen-prior", action="store_true", default=False, help="use frozen features for prior")
    parser.add_argument("--ttest-eval", action="store_true", default=False, help="evaluate the instance-level confidence scores")
    parser.add_argument("--viz-module-selection", action="store_true", default=False, help="visualize module selection trend")

    parser.add_argument("--fscil", action="store_true", default=False, help="enable few-shot CIL setting")
    parser.add_argument("--base-task-cls", type=int, default=60, help='num of classes in base task')
    parser.add_argument("--k-shot", type=int, default=5, help='num of training images per class')


    args, unparsed = parser.parse_known_args()
    args.mean_per_class = False

    if args.ckpt_path is None:
       args.ckpt_path = 'ckpt/{}.pt'.format(args.arch)

    args.save_path = args.save_path + '/' + args.db_name
    args.seed = args.num_run

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(args.seed)
    args.test_batch = args.train_batch 

    if args.compute_ram:
        args.ram_computer = RotationAngleMatrix(args)

    if args.model == 'clclip':
        args.method = "no_replay"
        model = ZeroshotCLIP(args)
    elif args.model == 'clclip_var':
        model = ClClipVariational(args)
    elif args.model == "clip_adapter":
        model = ClipAdapter(args)
    else:
        raise NotImplementedError

    if args.db_name == "cifar100":
        args.memory_type = "fix_total"
        args.memory = 2000
    else:
        args.memory_type = "fix_per_cls"
        if args.db_name in ["imagenet-r"]:
            args.num_class = 200
            args.class_per_task = 20
    args.num_task = args.num_class // args.class_per_task
    if args.exemplar_selector == 'random':
        selector = RandomExemplarsSelector(args)
    elif args.exemplar_selector == 'icarl':
        selector = HerdingExemplarsSelector(args)
    elif args.exemplar_selector == 'entropy':
        selector = EntropyExemplarsSelector(args)
    elif args.exemplar_selector == 'variance':
        selector = VarianceExemplarsSelector(args)
    elif args.exemplar_selector == 'distance':
        selector = DistanceExemplarsSelector(args)
    elif args.exemplar_selector == "var_entropy":
        selector = VarianceEntropyExemplarsSelector(args)
    elif args.exemplar_selector == 'energy':
        selector = EnergyExemplarsSelector(args)
    else:
        raise NotImplementedError 

    if not os.path.isdir(args.ckpt_path):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.save_path):
        mkdir_p(args.save_path)
    np.save(args.checkpoint + "/seed.npy", args.seed)
    try:
        shutil.copy2('main_incremental_submit.py', args.checkpoint)
        shutil.copy2('./classifier/vcop_4.py', args.checkpoint)
    except:
        pass
    inc_dataset = incremental_dataloader.IncrementalDataset(
                        dataset_name=args.db_name,
                        args = args,
                        random_order=False, #random class
                        shuffle=True,
                        seed=args.seed,
                        batch_size=args.train_batch,
                        workers=8,
                        validation_split=0,
                        increment=args.class_per_task,
                        exemplar_selector = selector
                    )
    start_sess = args.start_sess
    memory = None
    
    ctx_vec=None
    print(args)
    for ses in range(start_sess,  args.num_task):
        if ses > args.start_sess:
            if "er" in args.method:
                memory = pickle.load(open(args.save_path + "/memory_"+str(args.sess)+".pickle", 'rb'))
        task_info, train_loader, class_name, test_class, test_loader, for_memory, ood_test_loader = inc_dataset.new_task(memory) 
        
        args.sess=ses   
      
        if(start_sess==ses and start_sess!=0): 
            inc_dataset._current_task = ses
            with open(args.save_path + "/sample_per_task_testing_"+str(args.sess-1)+".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing             
            
        print('ses:',ses)
        print(task_info)    
        print(inc_dataset.sample_per_task_testing)     # dict{task:len(test)}
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        len_train = task_info['n_train_data']

        prompt_templates = inc_dataset.prompt_templates
        data = {'train_loader': train_loader, 'class_names': class_name, 'prompt_templates': prompt_templates}
        
        model_fitted = model.fit(data)
        model.post_training(finalize=False)
        memory_loader = None
        
        if "er" in args.method:
            memory = inc_dataset.get_memory(model_fitted, memory, for_memory)
            if args.finetuning:
                memory_loader = inc_dataset.get_memory_loader(memory)
                data['memory_loader'] = memory_loader
        if ses > 0 and args.finetuning:
            model.finetuning(data)

        model.post_training(finalize=True)
        print('finish fit')
        
        acc = model.accuracy(test_loader, args.num_test, test_class, mean_per_class=args.mean_per_class, ood_test_loader=ood_test_loader)
        with open(args.save_path + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.save_path + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.save_path + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.viz_module_selection:
        with open(args.save_path + "/module_selection_trend.pickle", 'wb') as handle:
            pickle.dump(model.time_step_to_test_id_to_module_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':
    args = parse_option()
    main(args)