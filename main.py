import argparse

from trainer import BertGCNTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--bert_init', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--pretrained_bert_ckpt', default=None)
    parser.add_argument('--dataset', default='mr', choices=['mr', 'R8', 'R52', 'ohsumed', 'mr'])
    parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
    parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
    parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gcn_lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=1e-5)

    parser.add_argument('--finetune_bert', type=bool, default=False)
    parser.add_argument('--train_gcn', type=bool, default=True)

    args = parser.parse_args()

    if args.train_gcn:
        BertGCNTrainer(args)
    

if __name__ == '__main__':
    main()
    
