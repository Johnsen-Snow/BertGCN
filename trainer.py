import torch as th
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
import os
import sys
import logging
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
from build_graph import BuildGraph
from utils import normalize_adj

class BertGCNTrainer:
    def __init__(self, args):
        global model, ckpt_dir
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        if args.checkpoint_dir is None:
            ckpt_dir = './checkpoint/{}_{}_{}'.format(args.bert_init, args.gcn_model, args.dataset)
        else:
            ckpt_dir = args.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.init(args)
        self.prepare_data(args.dataset)

        # instantiate model according to class number
        if args.gcn_model == 'gcn':
            model = BertGCN(nb_class=self.nb_class, pretrained_model=args.bert_init, m=args.m, gcn_layers=args.gcn_layers,
                            n_hidden=args.n_hidden, dropout=args.dropout)
        else:
            model = BertGAT(nb_class=self.nb_class, pretrained_model=args.bert_init, m=args.m, gcn_layers=args.gcn_layers,
                            heads=args.heads, n_hidden=args.n_hidden, dropout=args.dropout)

        if args.pretrained_bert_ckpt is not None:
            ckpt = th.load(args.pretrained_bert_ckpt, map_location=th.device('cuda:0'))
            model.bert_model.load_state_dict(ckpt['bert_model'])
            model.classifier.load_state_dict(ckpt['classifier'])
        
        self.train(args)

    def init(self, args):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(message)s'))
        sh.setLevel(logging.INFO)
        fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        fh.setLevel(logging.INFO)
        logger = logging.getLogger('training logger')
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        self.logger = logger

        logger.info('arguments:')
        logger.info(str(args))
        logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

    def prepare_data(self, dataset):
        # Data Preprocess
        self.bd = BuildGraph(dataset)
        self.adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = self.bd.load_corpus()
        nb_node = features.shape[0]
        nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        # transform one-hot label to class ID for pytorch computation
        y = y_train + y_test + y_val
        self.y_train = y_train.argmax(axis=1)
        self.y = y.argmax(axis=1)

        # document mask used for update feature
        global doc_mask
        doc_mask  = train_mask + val_mask + test_mask
        self.nb_word = nb_node - nb_train - nb_val - nb_test
        self.nb_class = y_train.shape[1]

        self.nb_test = nb_test
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_node = nb_node
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask


    def get_dataloader(self, model, max_length, batch_size):
        # load documents and compute input encodings
        text = self.bd.content
        nb_test = self.nb_test
        nb_word = self.nb_word
        nb_train = self.nb_train
        nb_val = self.nb_val
        nb_node = self.nb_node

        def encode_input(text, tokenizer):
            input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            return input.input_ids, input.attention_mask


        input_ids, attention_mask = encode_input(text, model.tokenizer)
        self.input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
        self.attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

        # create index loader
        train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
        val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
        test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
        doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

        idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
        idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
        idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
        idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

        return idx_loader_train, idx_loader_val, idx_loader_test, idx_loader

    def get_graph(self):
        global g, model
        adj_norm = normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
        g.ndata['input_ids'], g.ndata['attention_mask'] = self.input_ids, self.attention_mask
        g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
            th.LongTensor(self.y), th.FloatTensor(self.train_mask), th.FloatTensor(self.val_mask), th.FloatTensor(self.test_mask)
        g.ndata['label_train'] = th.LongTensor(self.y_train)
        g.ndata['cls_feats'] = th.zeros((self.nb_node, model.feat_dim))

        self.logger.info('graph information:')
        self.logger.info(str(g))

        return g

    def train(self, args):
        global model, optimizer
        idx_loader_train, idx_loader_val, idx_loader_test, idx_loader = self.get_dataloader(model, args.max_length, args.batch_size)
        self.get_graph()
        optimizer = th.optim.Adam([
                {'params': model.bert_model.parameters(), 'lr': args.bert_lr},
                {'params': model.classifier.parameters(), 'lr': args.bert_lr},
                {'params': model.gcn.parameters(), 'lr': args.gcn_lr},
            ], lr=1e-3
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

        trainer = Engine(train_step)
        pbar = ProgressBar()
        pbar.attach(trainer, ['loss'])

        evaluator = Engine(test_step)
        metrics={
            'acc': Accuracy(),
            'nll': Loss(th.nn.NLLLoss())
        }
        for n, f in metrics.items():
            f.attach(evaluator, n)

        @trainer.on(Events.EPOCH_COMPLETED)
        def reset_graph(trainer):
            scheduler.step()
            update_feature()
            th.cuda.empty_cache()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            global model, optimizer
            evaluator.run(idx_loader_train)
            metrics = evaluator.state.metrics
            train_acc, train_nll = metrics["acc"], metrics["nll"]
            evaluator.run(idx_loader_val)
            metrics = evaluator.state.metrics
            val_acc, val_nll = metrics["acc"], metrics["nll"]
            evaluator.run(idx_loader_test)
            metrics = evaluator.state.metrics
            test_acc, test_nll = metrics["acc"], metrics["nll"]
            self.logger.info(
                "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
                .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
            )
            if val_acc > log_training_results.best_val_acc:
                self.logger.info("New checkpoint")
                th.save(
                    {
                        'bert_model': model.bert_model.state_dict(),
                        'classifier': model.classifier.state_dict(),
                        'gcn': model.gcn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': trainer.state.epoch,
                    },
                    os.path.join(
                        ckpt_dir, 'checkpoint.pth'
                    )
                )
                log_training_results.best_val_acc = val_acc

        log_training_results.best_val_acc = 0
        g = update_feature()
        trainer.run(idx_loader, max_epochs=args.nb_epochs)

# Training
def update_feature():
    global model, g, doc_mask
    cpu, gpu = th.device('cpu'), th.device('cuda:0')
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g

def train_step(engine, batch):
    global model, g, optimizer
    cpu, gpu = th.device('cpu'), th.device('cuda:0')
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


def test_step(engine, batch):
    global model, g
    gpu = th.device('cuda:0')
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true
