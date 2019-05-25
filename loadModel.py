from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField
from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.predictors.predictor import Predictor

import torch.optim as optim
import numpy as np

import json
import pickle
import torch

train_input = pickle.load(open("mlinput_merge_with_entity.txt", 'rb'))
test = json.load(open('text_result2.0.json', 'r+'))
idlist=[]

class ProjDataSetReader(DatasetReader):

    def __init__(self, sentence_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.sentence_indexers = sentence_indexers or {"sentence": SingleIdTokenIndexer()}

    def text_to_instance(self, claim:str,evidence:List[str],label:str=None)->Instance:
        tobeField=[]
        for claim_word in claim.split(' '):
            tobeField.append(Token(claim_word))
        claim_field=TextField(tobeField,self.sentence_indexers)
        tobeField=[]
        for evidence_sent in evidence:
            for evidence_word in evidence_sent.split(' '):
                tobeField.append(Token(evidence_word))
        evidence_field=TextField(tobeField,self.sentence_indexers)
        if label!=None:
            fields={'claim':claim_field,'evidence':evidence_field,'label':LabelField(label)}
        else:
            fields={'claim':claim_field,'evidence':evidence_field}
        return Instance(fields)

    def _read(self, file_path: str)->Iterator[Instance]:
        if file_path=="train":
            pre = ""
            for value in train_input:
                claim = value['claim']
                if 'label' in value:
                    label = value['label']
                else:
                    label=None
                if label == "NOT ENOUGH INFO":
                #                     continue
                #instance = claim + ' ' + pre
                    evidence=[pre]
                    yield self.text_to_instance(claim,evidence,label)
                else:
                    i = 0
                    full_evidence=[]
                    for e in value['evidence']:
                        pre = ' '.join(e)
                    #instance += ' ' + pre
                        full_evidence.append(pre)
                        i += 1
                        if i >= 3:
                            break
                    yield self.text_to_instance(claim,full_evidence,label)
        elif file_path=='test':
            for id in test:
                idlist.append(id)
                claim=test[id]['claim']
                evidence=test[id]['evidence']
                if len(evidence)>0:
                    i=0
                    full_evidence=[]
                    for e in evidence:
                        full_evidence.append(e[2])
                        i+=1
                        if i>=3:
                            break
                else:
                     full_evidence=[]
                yield self.text_to_instance(claim,full_evidence)

class Lstm(Model):
    def __init__(self,
                 word_embedding:TextFieldEmbedder,
                 encoder:Seq2VecEncoder,
                 vocab:Vocabulary)->None:
        super().__init__(vocab)
        self.word_embedding=word_embedding
        self.encoder=encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                claim:Dict[str,torch.Tensor],
                evidence:Dict[str,torch.Tensor],
                label=None)->Dict[str,torch.Tensor]:
        claim_mask=get_text_field_mask(claim)
        claim_embedding=self.word_embedding(claim)
        claim_encoder_out=self.encoder(claim_embedding,claim_mask)
        claim_tag_logits=self.hidden2tag(claim_encoder_out)
        evidence_mask=get_text_field_mask(evidence)
        evidence_embedding=self.word_embedding(evidence)
        evidence_encoder_out=self.encoder(evidence_embedding,evidence_mask)
        evidence_tag_logits=self.hidden2tag(evidence_encoder_out)
        tag_logits=claim_tag_logits+evidence_tag_logits
        output={'tag_logits':tag_logits}
        if label is not None:
            self.accuracy(tag_logits,label)
            loss=torch.nn.CrossEntropyLoss()
            output['loss']=loss(tag_logits,label)
        return output

    def get_metrics(self, reset: bool = False)->Dict[str,float]:
        return {'accuracy':self.accuracy.get_metric(reset)}

reader = ProjDataSetReader()

EMBEDDING_DIM = 300
HIDDEN_DIM = 6

vocab=Vocabulary.from_files("vocabulary5")

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"sentence": token_embedding})

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model=Lstm(word_embeddings, lstm, vocab)

with open("model5.th", 'rb') as f:
    model.load_state_dict(torch.load(f,map_location='cpu'))

print("start reading test dataset")
test_dataset=reader.read('test')
print("reading finished")


iterator=BasicIterator()

iterator.index_with(vocab)

batch_generator = iterator(test_dataset, num_epochs=1, shuffle=False)
model.eval()
softmax=torch.nn.Softmax()

print("starting predict")
with torch.no_grad():
    i=0
    for batch in batch_generator:
        logits = model(**batch)
        ids = softmax(torch.FloatTensor(logits['tag_logits']))
        maxid = np.argmax(ids,axis=1)
        j=0
        for id in maxid:
            if len(test[idlist[j+i*32]]['evidence'])!=0:
                if int(id)==0:
                    test[idlist[j+i*32]]['label']="SUPPORTS"
                elif int(id)==1:
                    test[idlist[j+i*32]]['label']="NOT ENOUGH INFO"
                elif int(id)==2:
                    test[idlist[j+i*32]]['label']="REFUTES"
            else:
                test[idlist[j + i * 32]]['label'] = "NOT ENOUGH INFO"
            if test[idlist[j + i * 32]]['label'] == "NOT ENOUGH INFO":
                test[idlist[j + i * 32]]['evidence'] = []
            else:
                for entry in test[idlist[j + i * 32]]['evidence']:
                    entry.pop(2)
            '''
            place the evidence here
            '''
            j+=1
        i+=1



with open('answer4.0.json','w+') as f:
    json.dump(test,f)
