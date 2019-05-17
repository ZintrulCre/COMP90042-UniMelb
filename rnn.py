from nltk.tokenize import word_tokenize
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField
from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator,BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor


import torch.optim as optim
import numpy as np





import pickle
import torch

class VerbDatasetReader(DatasetReader):

    def __init__(self,sentence_indexers:Dict[str,TokenIndexer]=None )-> None:
        super().__init__(lazy=False)
        self.sentence_indexers=sentence_indexers or {"sentence":SingleIdTokenIndexer()}

    def text_to_instance(self, sentence:List[List],labels:str = None)->Instance:
        sent_tokenized=[]
        for sent in sentence:
            for word in word_tokenize(sent):
                sent_tokenized.append(Token(word))
        sentence_field=TextField(sent_tokenized,self.sentence_indexers)
        fields={'sentence':sentence_field,'labels':LabelField(labels)}
        return Instance(fields)

    def _read(self, file_path: str)->Iterator[Instance]:
        mlinput_merge=pickle.load(open(file_path,'rb'))
        for entry in mlinput_merge[:2000]:
            sentence_input=[entry['claim']]
            for sent in entry['evidence']:
                full_sent=' '.join(sent)
                sentence_input.append(full_sent)
            yield self.text_to_instance(sentence_input,entry['label'])

class Lstm(Model):
    def __init__(self,
                 word_embeddings:TextFieldEmbedder,
                 encoder:Seq2SeqEncoder,
                 vocab:Vocabulary)->None:
        super().__init__(vocab)
        self.word_embeddings=word_embeddings
        self.encoder=encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence:Dict[str,torch.Tensor],
                labels:torch.Tensor==None)->Dict[str,torch.Tensor]:
        mask=get_text_field_mask(sentence)
        print(len(sentence),len(labels))
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

reader=VerbDatasetReader()


print("start reading data")
train_dataset=reader.read('mlinput_merge.txt')
print("data reading finished")

vocab=Vocabulary.from_instances(train_dataset)
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('sentence'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"sentence": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = Lstm(word_embeddings, lstm, vocab)

optimizer = optim.SGD(model.parameters(), lr=0.1)

#iterator = BucketIterator(batch_size=64, sorting_keys=[("evidence", "num_tokens")])
iterator=BasicIterator()

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=train_dataset,
                  patience=10,
                  num_epochs=3)
print("start training model")
trainer.train()
print("training finished")

print("start saving the weights")
with open("test.th", 'wb') as f:
    torch.save(model.state_dict(), f)
print("saving finished")
print("start saving vocab")
vocab.save_to_files("vocabulary")
print("saving finished")

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

print('start doing prediction')
tag_logits = predictor.predict(['The Khmer Empire was not weak.','The Khmer Empire , officially the Angkor Empire , the predecessor state to modern Cambodia -LRB- `` Kampuchea '' or `` Srok Khmer '' to the Khmer people -RRB- , was a powerful Hindu-Buddhist empire in Southeast Asia .'])['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

print("predicting finished")

