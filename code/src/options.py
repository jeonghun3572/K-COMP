import argparse
import os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # basic parameters
        self.parser.add_argument("--output_dir", type=str, default="./checkpoint/train")
        self.parser.add_argument("--model_save_dir", type=str, default="./checkpoint/train")
        self.parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/train")
        self.parser.add_argument("--dataset", default=[])
        self.parser.add_argument("--train_data", nargs="+", default=[])
        self.parser.add_argument("--eval_data", nargs="+", default=[])
        self.parser.add_argument("--test_data", nargs="+", default=[])
        self.parser.add_argument("--eval_datasets", nargs="+", default=[])
        self.parser.add_argument("--eval_datasets_dir", type=str, default="./")
        self.parser.add_argument("--model_path", type=str, default=None)
        self.parser.add_argument("--continue_training", action="store_true")
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--max_seq_length", type=int, default=131072)

        self.parser.add_argument("--chunk_length", type=int, default=256)
        self.parser.add_argument("--loading_mode", type=str, default="split")
        self.parser.add_argument("--lower_case", action="store_true")
        self.parser.add_argument("--sampling_coefficient", type=float, default=0.0)
        self.parser.add_argument("--augmentation", type=str, default="none")
        self.parser.add_argument("--prob_augmentation", type=float, default=0.0)

        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--rho", type=float, default=0.05)

        self.parser.add_argument("--contrastive_mode", type=str, default="moco")
        self.parser.add_argument("--queue_size", type=int, default=65536)
        self.parser.add_argument("--tensor_parallel_size", type=int, default=1)
        self.parser.add_argument("--temperature", type=float, default=0.01)
        self.parser.add_argument("--top_p", type=float, default=1.0)
        self.parser.add_argument("--momentum", type=float, default=0.999)
        self.parser.add_argument("--moco_train_mode_encoder_k", action="store_true")
        self.parser.add_argument("--eval_normalize_text", action="store_true")
        self.parser.add_argument("--norm_query", action="store_true")
        self.parser.add_argument("--norm_doc", action="store_true")
        self.parser.add_argument("--projection_size", type=int, default=768)
        self.parser.add_argument("--max_grad_norm", type=float, default=1)

        self.parser.add_argument("--ratio_min", type=float, default=0.1)
        self.parser.add_argument("--ratio_max", type=float, default=0.5)
        self.parser.add_argument("--score_function", type=str, default="dot")
        #TODO
        self.parser.add_argument("--retriever_model_id", type=str, default="bert-base-uncased")
        self.parser.add_argument("--pooling", type=str, default="average")
        self.parser.add_argument("--random_init", action="store_true")

        # dataset parameters
        self.parser.add_argument("--batch_size", default=100, type=int)
        self.parser.add_argument("--per_device_train_batch_size", default=100, type=int)
        self.parser.add_argument("--per_device_eval_batch_size", default=100, type=int)
        self.parser.add_argument("--per_device_test_batch_size", default=100, type=int)
        self.parser.add_argument("--total_steps", type=int, default=1000)
        self.parser.add_argument("--eval_steps", type=int, default=100)
        self.parser.add_argument("--warmup_steps", type=int, default=-1)
        self.parser.add_argument("--warmup_ratio", type=float, default=0.0)
        self.parser.add_argument("--num_train_epochs", type=int, default=100)

        self.parser.add_argument("--local_rank", type=int, default=-1)
        self.parser.add_argument("--main_port", type=int, default=10001)
        self.parser.add_argument("--seed", type=int, default=42)
        # training parameters
        self.parser.add_argument("--optim", type=str, default="adamw_hf")
        self.parser.add_argument("--lr_scheduler_type", type=str, default="linear")
        self.parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_min_ratio", type=float, default=0.0)
        self.parser.add_argument("--weight_decay", type=float, default=0.0)
        self.parser.add_argument("--adam_beta1", type=float, default=0.9, help="beta1")
        self.parser.add_argument("--adam_beta2", type=float, default=0.999, help="beta2")
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-6, help="eps")
        self.parser.add_argument("--log_freq", type=int, default=100)
        self.parser.add_argument("--log_name", type=str, default='run')
        self.parser.add_argument("--swap_model", type=str, default='')
        self.parser.add_argument("--style", type=str, default=None)
        self.parser.add_argument("--eval_freq", type=int, default=100)
        self.parser.add_argument("--save_freq", type=int, default=50000)
        self.parser.add_argument("--maxload", type=int, default=None)
        self.parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
        self.parser.add_argument("--contrastive_weight", type=float, default=1.0)
        self.parser.add_argument("--loss1_factor", type=float, default=1.0)
        self.parser.add_argument("--loss2_factor", type=float, default=1.0)

        # finetuning options
        self.parser.add_argument('--verbalizer', type=str, default="\nPlease write an answer based on the passage.\nAnswer:")
        self.parser.add_argument('--verbalizer_passage', type=str, default="Passage: ")
        self.parser.add_argument('--verbalizer_question', type=str, default="\n\nQuestion: ")
        self.parser.add_argument('--n_docs', type=int, default=5)
        self.parser.add_argument('--n_docs_eval', type=int, default=5)
        self.parser.add_argument('--shard_size', type=int, default=1)

        #passage retrieval
        self.parser.add_argument("--no_fp16", action="store_true")
        self.parser.add_argument("--n_subquantizers", type=int, default=0)
        self.parser.add_argument("--n_bits", type=int, default=8)
        self.parser.add_argument("--model_name_or_path", type=str)
        self.parser.add_argument("--question_maxlength", type=int, default=512)
        self.parser.add_argument("--passages_embeddings", type=str, default=None)
        self.parser.add_argument("--passages", type=str, default=None)
        self.parser.add_argument("--save_or_load_index", action="store_true")
        self.parser.add_argument("--indexing_batch_size", type=int, default=1000000)
        self.parser.add_argument('--data_impl', type=str, default='infer', choices=['lazy', 'cached', 'mmap', 'infer'])
        self.parser.add_argument('--mmap_warmup', action='store_true')
        self.parser.add_argument("--passage_maxlength", type=int, default=512)
        self.parser.add_argument("--world_size", type=int, default=1)
        self.parser.add_argument("--lora_r", type=int, default=8)
        self.parser.add_argument("--lora_alpha", type=int, default=16)
        self.parser.add_argument("--lora_dropout", type=float, default=0.05)
        self.parser.add_argument("--model_id", type=str)
        self.parser.add_argument("--faiss_path", type=str, default=None)
        self.parser.add_argument("--wandb_run_name", type=str, default=None)
        self.parser.add_argument("--wandb_run", type=bool, default=False)
        self.parser.add_argument("--save_result", type=bool, default=False)
        self.parser.add_argument("--chat", type=bool, default=False)
        self.parser.add_argument("--use_faiss_gpu", default=True)
        self.parser.add_argument("--num_gpus", type=str, default=-1)
        self.parser.add_argument("--verbose", default=False)
        self.parser.add_argument("--temperature_gold", type=float, default=1.0)
        self.parser.add_argument("--temperature_score", type=float, default=1.0)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.parser.add_argument("--num_beams", type=int, default=10)
        self.parser.add_argument("--wandb_proj", type=str, default="huggingface")
        self.parser.add_argument("--few_shot", type=int, default=0)
        # ra_truncate_broken_sents
        # ra_round_broken_sents


    def print_options(self, opt):
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, "models"))
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
