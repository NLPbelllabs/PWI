import os
import collections
from tqdm import tqdm
import h5py
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import AnswerTable,load_lxmert_qa, load_lxmert_from_sgg_and_lxmert_pretrain, load_lxmert_from_pretrain_noqa
from tasks.vqa_model import VQAModel,VQAModel_Attention
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

'''Functions'''
'''Replace objects in one category into another category'''
def a2b(value,tset):
    # without wordpiece
    return tset.symbolic_vocab.obj_id2word(value)
def b2c(value,tokenizer):
    # after wordpiece
    #return tokenizer.tokenize(value)
    # tokens_id after wordpiece
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value))
def shift(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]
def list_mul(a,b):
    return [a0*b0 for a0,b0 in zip(a,b)]

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    print('splits:',splits)
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset, args)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True,
        collate_fn=lambda x: x
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = args.get("valid_batch_size", 16)
            self.valid_tuple = get_data_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        print('#################################')
        print('args.output_attention:',args.output_attention)
        if args.output_attention == True:
            self.model = VQAModel_Attention(self.train_tuple.dataset.num_answers)
        else:
            self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.get("load_lxmert_pretrain", None) is not None:
            load_lxmert_from_pretrain_noqa(args.load_lxmert_pretrain, self.model)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            self.model.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        train_results = []
        report_every = args.get("report_every", 100)
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, batch in iter_wrapper(enumerate(loader)):
                ques_id, feats, boxes, sent, tags, target = zip(*batch)
                self.model.train()
                self.optim.zero_grad()

                target = torch.stack(target).cuda()
                logit = self.model(feats, boxes, sent, tags)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                train_results.append(pd.Series({"loss":loss.detach().mean().item()}))

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
                
                if i % report_every == 0 and i > 0:
                    print("Epoch: {}, Iter: {}/{}".format(epoch, i, len(loader)))
                    print("    {}\n~~~~~~~~~~~~~~~~~~\n".format(pd.DataFrame(train_results[-report_every:]).mean()))

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid and not args.get("special_test", False):
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            if epoch >= 5:
                self.save("Epoch{}".format(epoch))
            print(log_str, end='')
            print(args.output)


        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, batch in enumerate(tqdm(loader)):
            _ = list(zip(*batch))
            ques_id, feats, boxes, sent, tags = _[:5]#, target = zip(*batch)
            with torch.no_grad():
                #target = torch.stack(target).cuda()
                logit = self.model(feats, boxes, sent, tags)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
        

'''with answer filtering'''
def PWI_FixedReplace(vqa,
        eval_tuple,
        tset,
        tokenizer,
        replace_obj_id,
        use_category,
       ans_filter=None,
       replace_en =False,
       feat_filter_en=True,
       No_tags=True):
    
    quesid2ans = {}
    quesid2ans['prob'] = []
    quesid2ans['ques_id'] = []
    #quesid2ans['ans'] = []
    quesid2ans['category_ans'] = []
    quesid2ans['category_en'] = []
    quesid2ans['target'] = []
    quesid2ans['N_obj_pick'] = []
    quesid2ans['all_ans'] = []
    quesid2ans['sent'] = []
    quesid2ans['img_id'] = []
    quesid2ans['ques_type'] = []
    quesid2ans['multi_ans'] = []
    quesid2ans['logit'] = []
    quesid2ans['LM_ans'] = []
    quesid2ans['LM_ans_score'] = []
    quesid2ans['obj_labels'] = []
    quesid2ans['attr_labels'] = []
    quesid2ans['obj_confs'] = []
    quesid2ans['attr_confs'] = []
#     if use_category == food_obj_id:
#         ans_category = food_ans_id
#     elif use_category == animal_obj_id:
#         ans_category = animal_ans_id

    softmax_layer = nn.Softmax(dim=1)
    for i, batch in enumerate(tqdm(eval_tuple.loader)):
        if ans_filter == None:
            use_category_tmp = use_category
        else:
            use_category_tmp = ans_filter[i*args.batch_size:(i*args.batch_size+len(batch))]
        
        #print(use_category_tmp)
        _ = list(zip(*batch))
        ques_id, feats, boxes, sent, tags, target, obj_labels, obj_confs, attr_labels, attr_confs,img_id,ques_type,multi_ans = _

        obj_labels_original=()
        obj_labels_replace=()
        obj_pick=[]
        N_obj_pick=[]
        for ii,obj_tmp in enumerate(obj_labels):
            '''get obj ID'''
            if ans_filter == None:
                obj_tmp0=[1 if obj_pos in use_category_tmp else 0 for obj_pos in obj_tmp]
            else:
                obj_tmp0=[1 if obj_pos == use_category_tmp[ii] else 0 for obj_pos in obj_tmp]
            #print(obj_tmp0)
            obj_pick+=[obj_tmp0]
            N_obj_pick.append(obj_tmp0.count(1))
            #objj=[animal_obj_id[randint(len(animal_obj_id))] if obj_pos in food_obj_id else obj_pos for obj_pos in obj_tmp]
            '''replace food obj with animal obj, random selection'''
            '''
            objj=[b2c(a2b(replace_obj_id[randint(len(replace_obj_id))]))
                  if obj_pos in use_category 
                  else b2c(a2b(obj_pos))
                  for obj_pos in obj_tmp]
            '''
            objj=[b2c(replace_obj_id[use_category.index(obj_pos)],tokenizer)
                  if obj_pos in use_category
                  else b2c(a2b(obj_pos,tset),tokenizer)
                  for obj_pos in obj_tmp]
#             objj=[]
#             for obj_pos in obj_tmp:
#                 if obj_pos in use_category:
#                     replace_obj_id2=[x for x in replace_obj_id if x != obj_pos]
#                     objj+=[b2c(a2b(replace_obj_id2[randint(len(replace_obj_id2))]))]
#                 else:
#                     objj+=[b2c(a2b(obj_pos))]
            obj_labels_replace = obj_labels_replace + (objj,)
            '''without replacement, for test only'''
            objj=[b2c(a2b(obj_pos,tset),tokenizer)
                  for obj_pos in obj_tmp]
            obj_labels_original = obj_labels_original + (objj,)

        '''Solve wordpiece problem'''
        obj_labels_replace2 = []
        obj_pick2 = []
        for obj_ori,obj_replace,obj_pick_tmp in zip(obj_labels_original,obj_labels_replace,obj_pick):
            replace_tmp = []
            obj_pick_tmp2 = []
            for obj_1,obj_2,obj_3 in zip(obj_ori,obj_replace,obj_pick_tmp):
                #print('length:',len(obj_1),len(obj_2))
                for i in range(len(obj_1)):
                    obj_pick_tmp2.append(obj_3)
                if len(obj_1) == len(obj_2):
                    replace_tmp+=obj_2
                elif len(obj_1) > len(obj_2):
                    replace_tmp+=(obj_2*(len(obj_1)//len(obj_2)+1))[:len(obj_1)]
                    #replace_tmp+=obj_2*(len(obj_1)//len(obj_2))
                elif len(obj_1) < len(obj_2):
                    #replace_tmp+=(obj_2*(len(obj_1)//len(obj_2)+1))[:len(obj_1)]
                    replace_tmp+=obj_2[:len(obj_1)]
            obj_labels_replace2.append(replace_tmp)
            obj_pick2.append(obj_pick_tmp2)
            
        if No_tags == True:
            obj_pick3=[]
            for i in obj_pick2:
                obj_pick3.append([i2*0 for i2 in i ])
            obj_pick2 = obj_pick3
            
        tags2=()
        nroll=0
        for i1 in range(len(tags)):
            #print(i1,len(tags[i1][0]))
            #i1=0
            tokens = list_mul(tags[i1][0],obj_pick2[i1])#+list_mul(tags[i1][0],obj_pick2[i1])
            #tokens[::2] = list_mul(tags[i1][0],obj_pick2[i1])
            '''
            False: keep original category tags
            True: replace tags with another category
            '''
            if replace_en:
                tokens = list_mul(obj_labels_replace2[i1],obj_pick2[i1])
            else:
                tokens = list_mul(shift(tags[i1][0], nroll),obj_pick2[i1])
            #tokens=[0.]*len(tokens)
            masks = list_mul(tags[i1][1],obj_pick2[i1])#+list_mul(tags[i1][1],obj_pick2[i1])
            #masks[::2] = list_mul(tags[i1][1],obj_pick2[i1])
            #masks[1::2] = list_mul(shift(tags[i1][1], nroll),obj_pick2[i1])
            for i2 in range(len(tags[i1][2])):
                box = list_mul(tags[i1][2],obj_pick2[i1])#+list_mul(tags[i1][2],obj_pick2[i1])
                #box[::2] = list_mul(tags[i1][2],obj_pick2[i1])
                #box[1::2] = list_mul(shift(tags[i1][2], nroll),obj_pick2[i1])
            tags2=tags2+([tokens,masks,box,None,None],)

        '''remove unrelated features extracted from image'''
        #print(len(feats),feats[0].shape)
        feats_filtered = ()
        boxes_filtered = ()
        for i in range(len(feats)):
            feats_filtered=feats_filtered+(feats[i]*np.array(obj_pick[i])[:,np.newaxis],)
            boxes_filtered=boxes_filtered+(boxes[i]*np.array(obj_pick[i])[:,np.newaxis],)

        with torch.no_grad():
            #target = torch.stack(target).cuda()
            #logit = vqa.model(feats, boxes, sent, tags)
            if feat_filter_en:
                logit = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
            else:
                logit = vqa.model(feats, boxes, sent, tags2)
#             '''target ans enable'''
#             ans_list=[]
#             ans_list2=[]
#             ans_en=[]
#             for i2 in range(len(target)):
#                 ans_en_tmp=[]
#                 ans_tmp=[]
#                 ans_tmp2=[]
#                 for i,tmp in enumerate(target[i2]):
#                     if tmp>0:
#                         ans_tmp2.append(i)
#                         if i in ans_category:
#                             ans_en_tmp.append(1)
#                             ans_tmp.append(i)
#                         else:
#                             ans_en_tmp.append(0)
#                 ans_en.append(ans_en_tmp)
#                 ans_list.append(ans_tmp)
#                 ans_list2.append(ans_tmp2)
#             quesid2ans['category_ans']+=ans_list
#             quesid2ans['all_ans']+=ans_list2
#             quesid2ans['category_en']+=ans_en
            quesid2ans['N_obj_pick']+=N_obj_pick
            quesid2ans['target']+=target
            quesid2ans['sent']+=[i for i in sent]
            quesid2ans['img_id']+=[i for i in img_id]
            quesid2ans['ques_type']+=[i for i in ques_type]
            quesid2ans['multi_ans']+=[i for i in multi_ans]
            quesid2ans['prob']+=[list(logit[i,target[i]>0].cpu().numpy()) for i in range(len(target))]
            quesid2ans['obj_labels']+= obj_labels
            quesid2ans['attr_labels']+= attr_labels
            quesid2ans['obj_confs']+= obj_confs
            quesid2ans['attr_confs']+= attr_confs
            #quesid2ans['category_ans']+=[i for i,tmp in enumerate(target[0]) if tmp>0 and i in food_obj_id for i in range(len(target))]
            logit=softmax_layer(logit)
            quesid2ans['logit']+=list(logit.cpu().numpy())
            score, label = logit.max(1)
            for qid,s, l in zip(ques_id, score.cpu().numpy(),label.cpu().numpy()):
                ans = eval_tuple.dataset.label2ans[l]
                #quesid2ans[qid] = ans
                quesid2ans['ques_id'].append(qid)
                quesid2ans['LM_ans'].append(ans)            
                quesid2ans['LM_ans_score'].append(s)
    return quesid2ans

'''with answer filtering'''
def PWI_FixedReplace_QuesModify(quesfunc,
        vqa,
        eval_tuple,
        tset,
        tokenizer,
        replace_obj_id,
        use_category,
       ans_filter=None,
       replace_en =False,
       feat_filter_en=True,
       No_tags=True
       ):
    
    quesid2ans = {}
    quesid2ans['prob'] = []
    quesid2ans['ques_id'] = []
    #quesid2ans['ans'] = []
    quesid2ans['category_ans'] = []
    quesid2ans['category_en'] = []
    quesid2ans['target'] = []
    quesid2ans['N_obj_pick'] = []
    quesid2ans['all_ans'] = []
    quesid2ans['sent'] = []
    quesid2ans['img_id'] = []
    quesid2ans['ques_type'] = []
    quesid2ans['multi_ans'] = []
    quesid2ans['logit'] = []
    quesid2ans['LM_ans'] = []
    quesid2ans['LM_ans_score'] = []
    quesid2ans['obj_labels'] = []
    quesid2ans['attr_labels'] = []
    quesid2ans['obj_confs'] = []
    quesid2ans['attr_confs'] = []
#     if use_category == food_obj_id:
#         ans_category = food_ans_id
#     elif use_category == animal_obj_id:
#         ans_category = animal_ans_id

    softmax_layer = nn.Softmax(dim=1)
    for i, batch in enumerate(tqdm(eval_tuple.loader)):
        if ans_filter == None:
            use_category_tmp = use_category
        else:
            use_category_tmp = ans_filter[i*args.batch_size:(i*args.batch_size+len(batch))]
        
        #print(use_category_tmp)
        _ = list(zip(*batch))
        ques_id, feats, boxes, sent, tags, target, obj_labels, obj_confs, attr_labels, attr_confs,img_id,ques_type,multi_ans = _

        sent = quesfunc(sent)

        obj_labels_original=()
        obj_labels_replace=()
        obj_pick=[]
        N_obj_pick=[]
        for ii,obj_tmp in enumerate(obj_labels):
            '''get obj ID'''
            if ans_filter == None:
                obj_tmp0=[1 if obj_pos in use_category_tmp else 0 for obj_pos in obj_tmp]
            else:
                obj_tmp0=[1 if obj_pos == use_category_tmp[ii] else 0 for obj_pos in obj_tmp]
            #print(obj_tmp0)
            obj_pick+=[obj_tmp0]
            N_obj_pick.append(obj_tmp0.count(1))
            #objj=[animal_obj_id[randint(len(animal_obj_id))] if obj_pos in food_obj_id else obj_pos for obj_pos in obj_tmp]
            '''replace food obj with animal obj, random selection'''
            '''
            objj=[b2c(a2b(replace_obj_id[randint(len(replace_obj_id))]))
                  if obj_pos in use_category 
                  else b2c(a2b(obj_pos))
                  for obj_pos in obj_tmp]
            '''
            objj=[b2c(replace_obj_id[use_category.index(obj_pos)],tokenizer)
                  if obj_pos in use_category
                  else b2c(a2b(obj_pos,tset),tokenizer)
                  for obj_pos in obj_tmp]
#             objj=[]
#             for obj_pos in obj_tmp:
#                 if obj_pos in use_category:
#                     replace_obj_id2=[x for x in replace_obj_id if x != obj_pos]
#                     objj+=[b2c(a2b(replace_obj_id2[randint(len(replace_obj_id2))]))]
#                 else:
#                     objj+=[b2c(a2b(obj_pos))]
            obj_labels_replace = obj_labels_replace + (objj,)
            '''without replacement, for test only'''
            objj=[b2c(a2b(obj_pos,tset),tokenizer)
                  for obj_pos in obj_tmp]
            obj_labels_original = obj_labels_original + (objj,)

        '''Solve wordpiece problem'''
        obj_labels_replace2 = []
        obj_pick2 = []
        for obj_ori,obj_replace,obj_pick_tmp in zip(obj_labels_original,obj_labels_replace,obj_pick):
            replace_tmp = []
            obj_pick_tmp2 = []
            for obj_1,obj_2,obj_3 in zip(obj_ori,obj_replace,obj_pick_tmp):
                #print('length:',len(obj_1),len(obj_2))
                for i in range(len(obj_1)):
                    obj_pick_tmp2.append(obj_3)
                if len(obj_1) == len(obj_2):
                    replace_tmp+=obj_2
                elif len(obj_1) > len(obj_2):
                    replace_tmp+=(obj_2*(len(obj_1)//len(obj_2)+1))[:len(obj_1)]
                    #replace_tmp+=obj_2*(len(obj_1)//len(obj_2))
                elif len(obj_1) < len(obj_2):
                    #replace_tmp+=(obj_2*(len(obj_1)//len(obj_2)+1))[:len(obj_1)]
                    replace_tmp+=obj_2[:len(obj_1)]
            obj_labels_replace2.append(replace_tmp)
            obj_pick2.append(obj_pick_tmp2)
            
        if No_tags == True:
            obj_pick3=[]
            for i in obj_pick2:
                obj_pick3.append([i2*0 for i2 in i ])
            obj_pick2 = obj_pick3
            
        tags2=()
        nroll=0
        for i1 in range(len(tags)):
            #print(i1,len(tags[i1][0]))
            #i1=0
            tokens = list_mul(tags[i1][0],obj_pick2[i1])#+list_mul(tags[i1][0],obj_pick2[i1])
            #tokens[::2] = list_mul(tags[i1][0],obj_pick2[i1])
            '''
            False: keep original category tags
            True: replace tags with another category
            '''
            if replace_en:
                tokens = list_mul(obj_labels_replace2[i1],obj_pick2[i1])
            else:
                tokens = list_mul(shift(tags[i1][0], nroll),obj_pick2[i1])
            #tokens=[0.]*len(tokens)
            masks = list_mul(tags[i1][1],obj_pick2[i1])#+list_mul(tags[i1][1],obj_pick2[i1])
            #masks[::2] = list_mul(tags[i1][1],obj_pick2[i1])
            #masks[1::2] = list_mul(shift(tags[i1][1], nroll),obj_pick2[i1])
            for i2 in range(len(tags[i1][2])):
                box = list_mul(tags[i1][2],obj_pick2[i1])#+list_mul(tags[i1][2],obj_pick2[i1])
                #box[::2] = list_mul(tags[i1][2],obj_pick2[i1])
                #box[1::2] = list_mul(shift(tags[i1][2], nroll),obj_pick2[i1])
            tags2=tags2+([tokens,masks,box,None,None],)

        '''remove unrelated features extracted from image'''
        #print(len(feats),feats[0].shape)
        feats_filtered = ()
        boxes_filtered = ()
        for i in range(len(feats)):
            feats_filtered=feats_filtered+(feats[i]*np.array(obj_pick[i])[:,np.newaxis],)
            boxes_filtered=boxes_filtered+(boxes[i]*np.array(obj_pick[i])[:,np.newaxis],)

        with torch.no_grad():
            #target = torch.stack(target).cuda()
            #logit = vqa.model(feats, boxes, sent, tags)
            if feat_filter_en:
                logit = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
            else:
                logit = vqa.model(feats, boxes, sent, tags2)
#             '''target ans enable'''
#             ans_list=[]
#             ans_list2=[]
#             ans_en=[]
#             for i2 in range(len(target)):
#                 ans_en_tmp=[]
#                 ans_tmp=[]
#                 ans_tmp2=[]
#                 for i,tmp in enumerate(target[i2]):
#                     if tmp>0:
#                         ans_tmp2.append(i)
#                         if i in ans_category:
#                             ans_en_tmp.append(1)
#                             ans_tmp.append(i)
#                         else:
#                             ans_en_tmp.append(0)
#                 ans_en.append(ans_en_tmp)
#                 ans_list.append(ans_tmp)
#                 ans_list2.append(ans_tmp2)
#             quesid2ans['category_ans']+=ans_list
#             quesid2ans['all_ans']+=ans_list2
#             quesid2ans['category_en']+=ans_en
            quesid2ans['N_obj_pick']+=N_obj_pick
            quesid2ans['target']+=target
            quesid2ans['sent']+=[i for i in sent]
            quesid2ans['img_id']+=[i for i in img_id]
            quesid2ans['ques_type']+=[i for i in ques_type]
            quesid2ans['multi_ans']+=[i for i in multi_ans]
            quesid2ans['prob']+=[list(logit[i,target[i]>0].cpu().numpy()) for i in range(len(target))]
            quesid2ans['obj_labels']+= obj_labels
            quesid2ans['attr_labels']+= attr_labels
            quesid2ans['obj_confs']+= obj_confs
            quesid2ans['attr_confs']+= attr_confs
            #quesid2ans['category_ans']+=[i for i,tmp in enumerate(target[0]) if tmp>0 and i in food_obj_id for i in range(len(target))]
            logit=softmax_layer(logit)
            quesid2ans['logit']+=list(logit.cpu().numpy())
            score, label = logit.max(1)
            for qid,s, l in zip(ques_id, score.cpu().numpy(),label.cpu().numpy()):
                ans = eval_tuple.dataset.label2ans[l]
                #quesid2ans[qid] = ans
                quesid2ans['ques_id'].append(qid)
                quesid2ans['LM_ans'].append(ans)            
                quesid2ans['LM_ans_score'].append(s)
    return quesid2ans


def PWI_wordpiece(vqa,
                eval_tuple,
                tset,
                tokenizer,
                target_obj_id = None,
                target_ans_id = None,
                section = 'full',
                feat_filter_en=True,
                No_tags=True):

    quesid2ans = {}
    quesid2ans['prob'] = []
    quesid2ans['ques_id'] = []
    quesid2ans['ans'] = []
    quesid2ans['category_ans'] = []
    quesid2ans['category_en'] = []
    quesid2ans['target'] = []
    quesid2ans['N_obj_pick'] = []
    quesid2ans['N_ans_pick'] = []
    quesid2ans['all_ans'] = []
    quesid2ans['sent'] = []
    quesid2ans['img_id'] = []
    quesid2ans['ques_type'] = []
    quesid2ans['multi_ans'] = []
    quesid2ans['logit'] = []
    quesid2ans['LM_ans'] = []
    quesid2ans['LM_ans_score'] = []
    quesid2ans['obj_labels'] = []
    quesid2ans['attr_labels'] = []
    quesid2ans['obj_confs'] = []
    quesid2ans['attr_confs'] = []
    quesid2ans['attr_confs'] = []
    quesid2ans['obj_pick2'] = []


    softmax_layer = nn.Softmax(dim=1)
    for i, batch in enumerate(tqdm(eval_tuple.loader)):

        _ = list(zip(*batch))
        ques_id, feats, boxes, sent, tags, target, obj_labels, obj_confs, attr_labels, attr_confs,img_id,ques_type,multi_ans = _

        '''parameters'''
        obj_labels_original=()

        obj_pick=[]
        ans_pick=[]
        N_obj_pick=[]
        N_ans_pick=[]
        for obj_tmp,ans_tmp in zip(obj_labels,multi_ans):
            '''get obj ID'''
            obj_tmp0=[1 if i in target_obj_id else 0 for i in obj_tmp]
            obj_pick+=[obj_tmp0]
            N_obj_pick.append(obj_tmp0.count(1))
            '''get ans ID'''
            idid = [i for i,i2 in enumerate(obj_tmp0) if i2==1]
            ans_tmp0 = [obj_tmp[i2] for i2 in idid]
            ans_tmp1 =[tset.symbolic_vocab.objects[i] for i in ans_tmp0]
            ans_tmp2=[1 if i in ans_tmp1 else 0 for i in ans_tmp]
            ans_pick+=[ans_tmp2]
            N_ans_pick.append(ans_tmp2.count(1))
            #print(obj_tmp0)

            '''original obj list'''
            objj=[b2c(a2b(obj_pos,tset),tokenizer) for obj_pos in obj_tmp]
            obj_labels_original = obj_labels_original + (objj,)

        '''Solve wordpiece problem'''
        obj_pick2 = []
        for obj_ori,obj_pick_tmp in zip(obj_labels_original,obj_pick):
            obj_pick_wp = []
            for obj_1,obj_2 in zip(obj_ori,obj_pick_tmp):
                if section ==  'full':
                    obj_pick_wp+=[obj_2 for i in range(len(obj_1)) ]
                elif section ==  'first':
                    obj_pick_wp+=[obj_2 if i==0 else 0 for i in range(len(obj_1)) ]
                elif section ==  'second':
                    obj_pick_wp+=[obj_2 if i==len(obj_1)-1 else 0 for i in range(len(obj_1)) ]
            obj_pick2.append(obj_pick_wp)
            
        if No_tags == True:
            obj_pick3=[]
            for i in obj_pick2:
                obj_pick3.append([i2*0 for i2 in i ])
            obj_pick2 = obj_pick3

            
        tags2=()
        nroll=0
          
        for i1 in range(len(tags)):
            #print(i1,len(tags[i1][0]))
            #i1=0
            tokens = list_mul(tags[i1][0],obj_pick2[i1])
            #tokens=[0.]*len(tokens)
            masks = list_mul(tags[i1][1],obj_pick2[i1])
            # masks=[i*0 for i in masks]
            for i2 in range(len(tags[i1][2])):
                box = list_mul(tags[i1][2],obj_pick2[i1])
            tags2=tags2+([tokens,masks,box,None,None],)

        '''remove unrelated features extracted from image'''
        #print(len(feats),feats[0].shape)
        feats_filtered = ()
        boxes_filtered = ()
        for i in range(len(feats)):
            feats_filtered=feats_filtered+(feats[i]*np.array(obj_pick[i])[:,np.newaxis],)
            boxes_filtered=boxes_filtered+(boxes[i]*np.array(obj_pick[i])[:,np.newaxis],)

        with torch.no_grad():
            #target = torch.stack(target).cuda()
            #logit = vqa.model(feats, boxes, sent, tags)
            if feat_filter_en:
                logit = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
            else:
                logit = vqa.model(feats, boxes, sent, tags2)
            '''target ans enable'''
            '''ans in the wordpiece list'''
            ans_list=[]
            '''all ans'''
            ans_list2=[]
            ans_en=[]
            for i2 in range(len(target)):
                ans_en_tmp=[]
                ans_tmp=[]
                ans_tmp2=[]
                for i,tmp in enumerate(target[i2]):
                    if tmp>0:
                        ans_tmp2.append(i)
                        if i in target_ans_id:
                            ans_en_tmp.append(1)
                            ans_tmp.append(i)
                        else:
                            ans_en_tmp.append(0)
                ans_en.append(ans_en_tmp)
                ans_list.append(ans_tmp)
                ans_list2.append(ans_tmp2)

            quesid2ans['obj_pick2']+= obj_pick2
            quesid2ans['category_ans']+=ans_list
            quesid2ans['all_ans']+=ans_list2
            quesid2ans['category_en']+=ans_en
            quesid2ans['N_obj_pick']+=N_obj_pick
            quesid2ans['N_ans_pick']+=N_ans_pick
            quesid2ans['target']+=target
            quesid2ans['sent']+=[i for i in sent]
            quesid2ans['img_id']+=[i for i in img_id]
            quesid2ans['ques_type']+=[i for i in ques_type]
            quesid2ans['multi_ans']+=[i for i in multi_ans]
            quesid2ans['obj_labels']+= obj_labels
            quesid2ans['attr_labels']+= attr_labels
            quesid2ans['obj_confs']+= obj_confs
            quesid2ans['attr_confs']+= attr_confs
            logit=softmax_layer(logit)
            quesid2ans['prob']+=[list(logit[i,target[i]>0].cpu().numpy()) for i in range(len(target))]
            quesid2ans['logit']+=list(logit.cpu().numpy())
            #quesid2ans['category_ans']+=[i for i,tmp in enumerate(target[0]) if tmp>0 and i in food_obj_id for i in range(len(target))]
            score, label = logit.max(1)

            for qid,s, l in zip(ques_id, score.cpu().numpy(),label.cpu().numpy()):
                ans = eval_tuple.dataset.label2ans[l]
                #quesid2ans[qid] = ans
                quesid2ans['ques_id'].append(qid)
                quesid2ans['LM_ans'].append(ans)            
                quesid2ans['LM_ans_score'].append(s)
    return quesid2ans



def PWI_FixedReplace_QuesModify_Embed(quesfunc,
        vqa,
        eval_tuple,
        tset,
        tokenizer,
        replace_obj_id,
        use_category,
       ans_filter=None,
       replace_en =False,
       feat_filter_en=True,
       No_tags=True):
    
    quesid2ans = {}
    quesid2ans['prob'] = []
    quesid2ans['ques_id'] = []
    #quesid2ans['ans'] = []
    quesid2ans['category_ans'] = []
    quesid2ans['category_en'] = []
    quesid2ans['target'] = []
    quesid2ans['N_obj_pick'] = []
    quesid2ans['all_ans'] = []
    quesid2ans['sent'] = []
    quesid2ans['img_id'] = []
    quesid2ans['ques_type'] = []
    quesid2ans['multi_ans'] = []
    quesid2ans['logit'] = []
    quesid2ans['LM_ans'] = []
    quesid2ans['LM_ans_score'] = []
    quesid2ans['obj_labels'] = []
    quesid2ans['attr_labels'] = []
    quesid2ans['obj_confs'] = []
    quesid2ans['attr_confs'] = []

    quesid2ans['image_feats_bert'] = ()
    quesid2ans['image_tags_bert'] = ()
    quesid2ans['image_obj'] = ()
    quesid2ans['N_obj_pick2'] = []
#     if use_category == food_obj_id:
#         ans_category = food_ans_id
#     elif use_category == animal_obj_id:
#         ans_category = animal_ans_id

    softmax_layer = nn.Softmax(dim=1)
    for i, batch in enumerate(tqdm(eval_tuple.loader)):
        if ans_filter == None:
            use_category_tmp = use_category
        else:
            use_category_tmp = ans_filter[i*args.batch_size:(i*args.batch_size+len(batch))]
        
        #print(use_category_tmp)
        _ = list(zip(*batch))
        ques_id, feats, boxes, sent, tags, target, obj_labels, obj_confs, attr_labels, attr_confs,img_id,ques_type,multi_ans = _
        #print(ques_id)
        sent = quesfunc(sent)

        obj_labels_original=()
        obj_labels_replace=()
        obj_pick=[]
        N_obj_pick=[]
        N_obj_pick2=[]
        for ii,obj_tmp in enumerate(obj_labels):
            '''get obj ID'''
            if ans_filter == None:
                obj_tmp0=[1 if obj_pos in use_category_tmp else 0 for obj_pos in obj_tmp]
            else:
                obj_tmp0=[1 if obj_pos == use_category_tmp[ii] else 0 for obj_pos in obj_tmp]
            #print(obj_tmp0)
            obj_pick+=[obj_tmp0]
            N_obj_pick.append(obj_tmp0.count(1))
            #objj=[animal_obj_id[randint(len(animal_obj_id))] if obj_pos in food_obj_id else obj_pos for obj_pos in obj_tmp]
            '''replace food obj with animal obj, random selection'''
            '''
            objj=[b2c(a2b(replace_obj_id[randint(len(replace_obj_id))]))
                  if obj_pos in use_category 
                  else b2c(a2b(obj_pos))
                  for obj_pos in obj_tmp]
            '''
            objj=[b2c(replace_obj_id[use_category.index(obj_pos)],tokenizer)
                  if obj_pos in use_category
                  else b2c(a2b(obj_pos,tset),tokenizer)
                  for obj_pos in obj_tmp]
#             objj=[]
#             for obj_pos in obj_tmp:
#                 if obj_pos in use_category:
#                     replace_obj_id2=[x for x in replace_obj_id if x != obj_pos]
#                     objj+=[b2c(a2b(replace_obj_id2[randint(len(replace_obj_id2))]))]
#                 else:
#                     objj+=[b2c(a2b(obj_pos))]
            #print('a:',(objj[0]))
            obj_labels_replace = obj_labels_replace + (objj,)
            '''without replacement, for test only'''
            objj=[b2c(a2b(obj_pos,tset),tokenizer)
                  for obj_pos in obj_tmp]
            #print('b:',(objj[0]))
            obj_labels_original = obj_labels_original + (objj,)
        #print('a0:',len(obj_labels_original),len(obj_labels_original[0]))
        #print('b0:',len(obj_pick),len(obj_pick[0]))

        '''Solve wordpiece problem'''
        obj_labels_replace2 = []
        obj_pick2 = []
        for obj_ori,obj_replace,obj_pick_tmp in zip(obj_labels_original,obj_labels_replace,obj_pick):
            replace_tmp = []
            obj_pick_tmp2 = []
            for obj_1,obj_2,obj_3 in zip(obj_ori,obj_replace,obj_pick_tmp):
                #print('length:',len(obj_1),len(obj_2),obj_3)
                '''extend obj_pick (either 0 or 1) with respect to the length of words after tokenizer'''
                for i in range(len(obj_1)):
                    obj_pick_tmp2.append(obj_3)
                '''tailor taken IDs to match the length of original words'''
                if len(obj_1) == len(obj_2):
                    replace_tmp+=obj_2
                elif len(obj_1) > len(obj_2):
                    replace_tmp+=(obj_2*(len(obj_1)//len(obj_2)+1))[:len(obj_1)]
                    #replace_tmp+=obj_2*(len(obj_1)//len(obj_2))
                elif len(obj_1) < len(obj_2):
                    #replace_tmp+=(obj_2*(len(obj_1)//len(obj_2)+1))[:len(obj_1)]
                    replace_tmp+=obj_2[:len(obj_1)]
            obj_labels_replace2.append(replace_tmp)
            obj_pick2.append(obj_pick_tmp2)
            N_obj_pick2.append(obj_pick_tmp2.count(1))
        #print('a1:',len(obj_labels_replace2),len(obj_labels_replace2[0]),len(obj_labels_replace2[1]))
        #print('b1:',len(obj_pick2),len(obj_pick2[0]),len(obj_pick2[1]))
        #print('c1:',len(obj_labels_replace),len(obj_labels_replace[0]),len(obj_labels_replace[1]))
        if No_tags == True:
            obj_pick3=[]
            for i in obj_pick2:
                obj_pick3.append([i2*0 for i2 in i ])
            obj_pick2 = obj_pick3
            
        tags2=()
        nroll=0
        for i1 in range(len(tags)):
            #print(i1,len(tags[i1][0]))
            #i1=0
            
            #tokens = list_mul(tags[i1][0],obj_pick2[i1])#+list_mul(tags[i1][0],obj_pick2[i1])
            #tokens[::2] = list_mul(tags[i1][0],obj_pick2[i1])
            '''
            False: keep original category tags
            True: replace tags with another category
            '''
            if replace_en:
                tokens = list_mul(obj_labels_replace2[i1],obj_pick2[i1])
            else:
                tokens = list_mul(shift(tags[i1][0], nroll),obj_pick2[i1])
            #tokens=[0.]*len(tokens)
            masks = list_mul(tags[i1][1],obj_pick2[i1])#+list_mul(tags[i1][1],obj_pick2[i1])
            #masks[::2] = list_mul(tags[i1][1],obj_pick2[i1])
            #masks[1::2] = list_mul(shift(tags[i1][1], nroll),obj_pick2[i1])
            for i2 in range(len(tags[i1][2])):
                box = list_mul(tags[i1][2],obj_pick2[i1])#+list_mul(tags[i1][2],obj_pick2[i1])
                #box[::2] = list_mul(tags[i1][2],obj_pick2[i1])
                #box[1::2] = list_mul(shift(tags[i1][2], nroll),obj_pick2[i1])
            tags2=tags2+([tokens,masks,box,None,None],)
            #print('a2:',len(tokens),len(box))
        '''remove unrelated features extracted from image'''
        #print(len(feats),feats[0].shape)
        feats_filtered = ()
        boxes_filtered = ()
        for i in range(len(feats)):
            feats_filtered=feats_filtered+(feats[i]*np.array(obj_pick[i])[:,np.newaxis],)
            boxes_filtered=boxes_filtered+(boxes[i]*np.array(obj_pick[i])[:,np.newaxis],)

        with torch.no_grad():
            #target = torch.stack(target).cuda()
            #logit = vqa.model(feats, boxes, sent, tags)
            #print(feats_filtered)
            if feat_filter_en:
                #logit = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
                logit, pooled_output, attention, stuff = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
            else:
                logit, pooled_output, attention, stuff = vqa.model(feats, boxes, sent, tags2)
            
            '''contextual embedding'''  
            image_feats = stuff[2].cpu().numpy()
            image_tags = stuff[1].cpu().numpy()
            #print(image_feats.shape,image_tags.shape)
            #print(len(obj_pick),len(obj_pick[0]))
            #print(len(obj_pick2),len(obj_pick2[0]))
            #image_feats_bert=[]
            #image_tags_bert=[]
            #image_obj=[]
            for inow,(feats1,pick1,tags1,pick2,obj1) in enumerate(zip(image_feats,obj_pick,image_tags,obj_pick2,obj_labels_replace2)):
                #image_feats_bert=[]
                #image_tags_bert=[]
                #image_obj=[]
                #print(inow,feats1.shape,len(pick1),tags1.shape,len(pick2))
                #print(inow,len([objs for objs,pick in zip(obj1,pick2) if pick==1]))
                image_feats_bert=[feats for feats,pick in zip(feats1,pick1) if pick==1]
                image_tags_bert=[tags for tags,pick in zip(tags1,pick2) if pick==1]
                image_obj=[objs for objs,pick in zip(obj1,pick2) if pick==1]
                quesid2ans['image_feats_bert']=quesid2ans['image_feats_bert']+(image_feats_bert,)
                quesid2ans['image_tags_bert']=quesid2ans['image_tags_bert']+(image_tags_bert,)
                quesid2ans['image_obj']=quesid2ans['image_obj']+(image_obj,)

            quesid2ans['N_obj_pick']+=N_obj_pick
            quesid2ans['N_obj_pick2']+=N_obj_pick2
            quesid2ans['target']+=target
            quesid2ans['sent']+=[i for i in sent]
            quesid2ans['img_id']+=[i for i in img_id]
            quesid2ans['ques_type']+=[i for i in ques_type]
            quesid2ans['multi_ans']+=[i for i in multi_ans]
            quesid2ans['prob']+=[list(logit[i,target[i]>0].cpu().numpy()) for i in range(len(target))]
            quesid2ans['obj_labels']+= obj_labels
            quesid2ans['attr_labels']+= attr_labels
            quesid2ans['obj_confs']+= obj_confs
            quesid2ans['attr_confs']+= attr_confs
            #quesid2ans['category_ans']+=[i for i,tmp in enumerate(target[0]) if tmp>0 and i in food_obj_id for i in range(len(target))]
            logit=softmax_layer(logit)
            quesid2ans['logit']+=list(logit.cpu().numpy())
            score, label = logit.max(1)
            for qid,s, l in zip(ques_id, score.cpu().numpy(),label.cpu().numpy()):
                ans = eval_tuple.dataset.label2ans[l]
                #quesid2ans[qid] = ans
                quesid2ans['ques_id'].append(qid)
                quesid2ans['LM_ans'].append(ans)            
                quesid2ans['LM_ans_score'].append(s)
    return quesid2ans



def PWI_wordpiece_Embed(vqa,
                eval_tuple,
                tset,
                tokenizer,
                target_obj_id = None,
                target_ans_id = None,
                section = 'full',
                feat_filter_en=True,
                No_tags=True):

    quesid2ans = {}
    quesid2ans['prob'] = []
    quesid2ans['ques_id'] = []
    quesid2ans['ans'] = []
    quesid2ans['category_ans'] = []
    quesid2ans['category_en'] = []
    quesid2ans['target'] = []
    quesid2ans['N_obj_pick'] = []
    quesid2ans['N_ans_pick'] = []
    quesid2ans['all_ans'] = []
    quesid2ans['sent'] = []
    quesid2ans['img_id'] = []
    quesid2ans['ques_type'] = []
    quesid2ans['multi_ans'] = []
    quesid2ans['logit'] = []
    quesid2ans['LM_ans'] = []
    quesid2ans['LM_ans_score'] = []
    quesid2ans['obj_labels'] = []
    quesid2ans['attr_labels'] = []
    quesid2ans['obj_confs'] = []
    quesid2ans['attr_confs'] = []
    quesid2ans['attr_confs'] = []
    quesid2ans['obj_pick2'] = []

    quesid2ans['image_feats_bert'] = ()
    quesid2ans['image_tags_bert'] = ()
    quesid2ans['image_obj'] = ()

    softmax_layer = nn.Softmax(dim=1)
    for i, batch in enumerate(tqdm(eval_tuple.loader)):

        _ = list(zip(*batch))
        ques_id, feats, boxes, sent, tags, target, obj_labels, obj_confs, attr_labels, attr_confs,img_id,ques_type,multi_ans = _

        '''parameters'''
        obj_labels_original=()

        obj_pick=[]
        ans_pick=[]
        N_obj_pick=[]
        N_ans_pick=[]
        for obj_tmp,ans_tmp in zip(obj_labels,multi_ans):
            '''get obj ID'''
            obj_tmp0=[1 if i in target_obj_id else 0 for i in obj_tmp]
            obj_pick+=[obj_tmp0]
            N_obj_pick.append(obj_tmp0.count(1))
            '''get ans ID'''
            idid = [i for i,i2 in enumerate(obj_tmp0) if i2==1]
            ans_tmp0 = [obj_tmp[i2] for i2 in idid]
            ans_tmp1 =[tset.symbolic_vocab.objects[i] for i in ans_tmp0]
            ans_tmp2=[1 if i in ans_tmp1 else 0 for i in ans_tmp]
            ans_pick+=[ans_tmp2]
            N_ans_pick.append(ans_tmp2.count(1))
            #print(obj_tmp0)

            '''original obj list'''
            objj=[b2c(a2b(obj_pos,tset),tokenizer) for obj_pos in obj_tmp]
            obj_labels_original = obj_labels_original + (objj,)

        '''Solve wordpiece problem'''
        obj_pick2 = []
        for obj_ori,obj_pick_tmp in zip(obj_labels_original,obj_pick):
            obj_pick_wp = []
            for obj_1,obj_2 in zip(obj_ori,obj_pick_tmp):
                if section ==  'full':
                    obj_pick_wp+=[obj_2 for i in range(len(obj_1)) ]
                elif section ==  'first':
                    obj_pick_wp+=[obj_2 if i==0 else 0 for i in range(len(obj_1)) ]
                elif section ==  'second':
                    obj_pick_wp+=[obj_2 if i==len(obj_1)-1 else 0 for i in range(len(obj_1)) ]
            obj_pick2.append(obj_pick_wp)
            
        if No_tags == True:
            obj_pick3=[]
            for i in obj_pick2:
                obj_pick3.append([i2*0 for i2 in i ])
            obj_pick2 = obj_pick3

            
        tags2=()
        nroll=0
          
        for i1 in range(len(tags)):
            #print(i1,len(tags[i1][0]))
            #i1=0
            tokens = list_mul(tags[i1][0],obj_pick2[i1])
            #tokens=[0.]*len(tokens)
            masks = list_mul(tags[i1][1],obj_pick2[i1])
            # masks=[i*0 for i in masks]
            for i2 in range(len(tags[i1][2])):
                box = list_mul(tags[i1][2],obj_pick2[i1])
            tags2=tags2+([tokens,masks,box,None,None],)

        '''remove unrelated features extracted from image'''
        #print(len(feats),feats[0].shape)
        feats_filtered = ()
        boxes_filtered = ()
        for i in range(len(feats)):
            feats_filtered=feats_filtered+(feats[i]*np.array(obj_pick[i])[:,np.newaxis],)
            boxes_filtered=boxes_filtered+(boxes[i]*np.array(obj_pick[i])[:,np.newaxis],)

        with torch.no_grad():
            #target = torch.stack(target).cuda()
            #logit = vqa.model(feats, boxes, sent, tags)
            if feat_filter_en:
                #logit = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
                logit, pooled_output, attention, stuff = vqa.model(feats_filtered, boxes_filtered, sent, tags2)
            else:
                logit, pooled_output, attention, stuff = vqa.model(feats, boxes, sent, tags2)

            '''contextual embedding'''  
            image_feats = stuff[2].cpu().numpy()
            image_tags = stuff[1].cpu().numpy()
            #print(image_feats.shape,image_tags.shape)
            #print(len(obj_pick),len(obj_pick[0]))
            #print(len(obj_pick2),len(obj_pick2[0]))
            #image_feats_bert=[]
            #image_tags_bert=[]
            #image_obj=[]
            for inow,(feats1,pick1,tags1,pick2) in enumerate(zip(image_feats,obj_pick,image_tags,obj_pick2)):
                #image_feats_bert=[]
                #image_tags_bert=[]
                #image_obj=[]
                #print(inow,feats1.shape,len(pick1),tags1.shape,len(pick2))
                #print(inow,len([objs for objs,pick in zip(obj1,pick2) if pick==1]))
                image_feats_bert=[feats for feats,pick in zip(feats1,pick1) if pick==1]
                image_tags_bert=[tags for tags,pick in zip(tags1,pick2) if pick==1]
                image_obj=[objs for objs,pick in zip(tags[inow][0],pick2) if pick==1]
                quesid2ans['image_feats_bert']=quesid2ans['image_feats_bert']+(image_feats_bert,)
                quesid2ans['image_tags_bert']=quesid2ans['image_tags_bert']+(image_tags_bert,)
                quesid2ans['image_obj']=quesid2ans['image_obj']+(image_obj,)


            quesid2ans['obj_pick2']+= obj_pick2
            #quesid2ans['category_ans']+=ans_list
            #quesid2ans['all_ans']+=ans_list2
            #quesid2ans['category_en']+=ans_en
            quesid2ans['N_obj_pick']+=N_obj_pick
            quesid2ans['N_ans_pick']+=N_ans_pick
            quesid2ans['target']+=target
            quesid2ans['sent']+=[i for i in sent]
            quesid2ans['img_id']+=[i for i in img_id]
            quesid2ans['ques_type']+=[i for i in ques_type]
            quesid2ans['multi_ans']+=[i for i in multi_ans]
            quesid2ans['obj_labels']+= obj_labels
            quesid2ans['attr_labels']+= attr_labels
            quesid2ans['obj_confs']+= obj_confs
            quesid2ans['attr_confs']+= attr_confs
            logit=softmax_layer(logit)
            quesid2ans['prob']+=[list(logit[i,target[i]>0].cpu().numpy()) for i in range(len(target))]
            quesid2ans['logit']+=list(logit.cpu().numpy())
            #quesid2ans['category_ans']+=[i for i,tmp in enumerate(target[0]) if tmp>0 and i in food_obj_id for i in range(len(target))]
            score, label = logit.max(1)

            for qid,s, l in zip(ques_id, score.cpu().numpy(),label.cpu().numpy()):
                ans = eval_tuple.dataset.label2ans[l]
                #quesid2ans[qid] = ans
                quesid2ans['ques_id'].append(qid)
                quesid2ans['LM_ans'].append(ans)            
                quesid2ans['LM_ans_score'].append(s)
    return quesid2ans
