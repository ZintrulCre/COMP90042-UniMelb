from nltk.tokenize import word_tokenize
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField
from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import pickle
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
        fields={'evidece':sentence_field,'label':LabelField(labels)}
        return Instance(fields)

    def _read(self, file_path: str)->Iterator[Instance]:
        mlinput_merge=pickle.load(open(file_path,'rb'))
        for entry in mlinput_merge:
            sentence_input=[entry['claim']]
            for sent in entry['evidence']:
                full_sent=' '.join(sent)
                sentence_input.append(full_sent)
            yield self.text_to_instance(sentence_input,entry['label'])



reader=VerbDatasetReader()

train_dataset=reader.read('mlinput_merge.txt')




