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
parser.add_argument('--lex_path', type=str, default="./ckpt/saved_ckpt/lex_sub.pt", help='pretrained lexical substituion model path')
parser.add_argument('--num_of_class', type=int, default=2,help='num_of_class')


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
from sa_model import SentimentClassifier
from subst_model import Subst_model
from sa_data_loader import get_s140_loader
from trainer import SA_Trainer
from tester import SA_Tester

CAND_MAX_LEN = 10



def main(args):
	tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")
	sentence_encoder = RobertaModel.from_pretrained("roberta-large", cache_dir="./ckpt/roberta")

	train_loader, pos_id2tag = get_s140_loader(tokenizer, split="train")
	test_loader, _ = get_s140_loader(tokenizer, split="test")

	if torch.cuda.is_available():	
		device = "cuda"
	else:
		device = "cpu"
	subst_model = Subst_model(sentence_encoder, tokenizer, device, candidate_max_length=CAND_MAX_LEN).to(device)
	subst_model.load_state_dict(torch.load(args.lex_path))
	sa_model = SentimentClassifier(encoder=sentence_encoder, tokenizer=tokenizer, num_class=args.num_of_class, 
								subst_generator=subst_model, pos_id2tag=pos_id2tag)
	sa_model.to(device)

	optimizer = torch.optim.Adam(sa_model.parameters(), lr=args.lr, betas=(0.9, 0.99))

	trainer = SA_Trainer(sa_model, train_loader, tokenizer, optimizer, test_loader, device, 
					  num_of_epo=args.epoch, dir_path=args.dir_path+"results/output", print_every=args.print_every)
	trainer.train()	

	tester = SA_Tester(sa_model, test_loader, args.dir_path, device, args.num_of_class)
	tester.test()



if __name__ == "__main__":
	
	
	main(args)