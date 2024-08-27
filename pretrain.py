import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ID', default='0', help='run ID')
parser.add_argument('--gpu', default='1', type=str, help='gpu device numbers')
parser.add_argument('--load_trained', type=bool, default=True, help='load trained model or not')

parser.add_argument('--epoch', type=int, default=100,help='random seed')
parser.add_argument('--batch_size', type=int, default=10,help='batch_size')
parser.add_argument('--num_workers', type=int, default=4,help='num_of_workers')
parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
parser.add_argument('--dir_path', type=str, default="EMNLP/", help='file path')
parser.add_argument('--alm_path', type=str, default="./ckpt/saved_ckpt/ALM.pt", help='pretrained ALM path')

parser.add_argument('--print_every', type=int, default=100,help='how many iter for averaging results')
parser.add_argument('--seed', type=int, default=666,help='random seed')
# parser.add_argument('--sent_max_length', type=int, default=60,help='sent_max_length')

args = parser.parse_args()

import os
# main_gpu = int(args.gpu.split(",")[0])
# torch.cuda.set_device(main_gpu)  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])

from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast
from trainer import Subst_Trainer
from tester import Subst_Tester
from subst_model import Subst_model
from lexsub_data_loader import get_lexsub_loader

CAND_MAX_LEN = 10



def main(args):
	tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")
	sentence_encoder = RobertaModel.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")

	lex_loader, test_loader = get_lexsub_loader(["coinco", "semeval_all"], "data", batch_size=args.batch_size, 
											 tokenizer=tokenizer, candidate_max_length=CAND_MAX_LEN, num_workers=args.num_workers)

	if torch.cuda.is_available():	
		device = "cuda"
	else:
		device = "cpu"
	subst_model = Subst_model(sentence_encoder, tokenizer, device, candidate_max_length=CAND_MAX_LEN).to(device)
	subst_model.load_state_dict(torch.load(args.alm_path))
	
	optimizer = torch.optim.Adam(subst_model.parameters(), lr=args.lr, betas=(0.9, 0.99))

	pretrainer = Subst_Trainer(subst_model, lex_loader, tokenizer, optimizer, test_loader, 
							device, num_of_epo=args.epoch, dir_path=args.dir_path+"pretrain/output", print_every=args.print_every)
	pretrainer.train()

	tester = Subst_Tester(subst_model, test_loader, args.dir_path+"pretrain/output", device)
	tester.test()



if __name__ == "__main__":
	
	
	main(args)