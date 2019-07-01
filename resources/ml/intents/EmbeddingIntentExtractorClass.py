import io
import logging
import numpy as np
import os
import pickle
import typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple
from pprint import pprint
import spacy


import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../')

from ..utils.TrainDataClass import TrainDataClass
from ..utils.FeatureExtractorClass import FeatureExtractorClass



logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    import tensorflow as tf

try:
    import tensorflow as tf
except ImportError:
    tf = None
    
    
class EmbeddingIntentExtractorClass():
    def __init__(self, featurizer: FeatureExtractorClass, dir_path: Text=None)->None:
        
        self.featurizer = featurizer
        self.INTENT_RANKING_LENGTH = 10

        self.defaults = {
                # nn architecture
                # sizes of hidden layers before the embedding layer for input words
                # the number of hidden layers is thus equal to the length of this list
                "hidden_layers_sizes_a": [256, 128],
                # sizes of hidden layers before the embedding layer for intent labels
                # the number of hidden layers is thus equal to the length of this list
                "hidden_layers_sizes_b": [],
                # training parameters
                # initial and final batch sizes - batch size will be
                # linearly increased for each epoch
                "batch_size": [64, 256],
                # number of epochs
                "epochs": 300,
                # embedding parameters
                # dimension size of embedding vectors
                "embed_dim": 20,
                # how similar the algorithm should try
                # to make embedding vectors for correct intent labels
                "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
                # maximum negative similarity for incorrect intent labels
                "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
                # the type of the similarity
                "similarity_type": "cosine",  # string 'cosine' or 'inner'
                # the number of incorrect intents, the algorithm will minimize
                # their similarity to the input words during training
                "num_neg": 20,
                # flag: if true, only minimize the maximum similarity for
                # incorrect intent labels
                "use_max_sim_neg": True,
                # set random seed to any int to get reproducible results
                # try to change to another int if you are not getting good results
                "random_seed": None,
                # regularization parameters
                # the scale of L2 regularization
                "C2": 0.002,
                # the scale of how critical the algorithm should be of minimizing the
                # maximum similarity between embeddings of different intent labels
                "C_emb": 0.8,
                # dropout rate for rnn
                "droprate": 0.2,
                # flag: if true, the algorithm will split the intent labels into tokens
                #       and use bag-of-words representations for them
                "intent_tokenization_flag": False,
                # delimiter string to split the intent labels
                "intent_split_symbol": "_",
                # visualization of accuracy
                # how often to calculate training accuracy
                "evaluate_every_num_epochs": 10,  # small values may hurt performance
                # how many examples to use for calculation of training accuracy
                "evaluate_on_num_examples": 1000,  # large values may hurt performance
            }
    
    def _load(self, file_name: Text="intent_model", model_dir: Text="intent_model")->None:
        self.model = self.load(file_name,model_dir)
    
    def _save(self, file_name: Text="intent_model", model_dir: Text="intent_model")->None:
        if self.model:
            self.persist(file_name=file_name,
                model_dir=model_dir,
                graph=self.model['graph'],
                session=self.model['session'],
                inv_intent_dict=self.model['inv_intent_dict'],
                encoded_all_intents=self.model['encoded_all_intents'],
                a_in=self.model['a_in'],
                b_in=self.model['b_in'],
                sim_op=self.model['sim_op'],
                word_embed=self.model['word_embed'],
                intent_embed=self.model['intent_embed']
               )
    
    def _train(self, data:TrainDataClass)->Dict:
        self.train_model(data)
        
    
    def _predict(self, text:Text)->Dict:
        intent_text_features = self.featurizer._get_text_features(text)
        if self.model:
            intetnts = self.process(
                message= intent_text_features,
                session=self.model['session'],
                inv_intent_dict=self.model['inv_intent_dict'],
                encoded_all_intents=self.model['encoded_all_intents'],
                a_in=self.model['a_in'],
                b_in=self.model['b_in'],
                sim_op=self.model['sim_op']
               )
            return intetnts
    
    
    
    # save model functions
    def persist(self, file_name: Text,
            model_dir: Text,
            graph: "tf.Graph",
            session: "tf.Session",
            inv_intent_dict: Dict,
            encoded_all_intents: np.ndarray,
            a_in: "tf.placeholder",
            b_in: "tf.placeholder",
            sim_op:"tf.Tensor",
            word_embed: "tf.Tensor",
            intent_embed: "tf.Tensor",
           ) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if session is None:
            return {"file": None}

        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise
        with graph.as_default():
            graph.clear_collection("message_placeholder")
            graph.add_to_collection("message_placeholder", a_in)

            graph.clear_collection("intent_placeholder")
            graph.add_to_collection("intent_placeholder", b_in)

            graph.clear_collection("similarity_op")
            graph.add_to_collection("similarity_op", sim_op)

            graph.clear_collection("word_embed")
            graph.add_to_collection("word_embed", word_embed)
            graph.clear_collection("intent_embed")
            graph.add_to_collection("intent_embed", intent_embed)

            saver = tf.train.Saver()
            saver.save(session, checkpoint)

        with io.open(
            os.path.join(model_dir, file_name + "_inv_intent_dict.pkl"), "wb"
        ) as f:
            pickle.dump(inv_intent_dict, f)
        with io.open(
            os.path.join(model_dir, file_name + "_encoded_all_intents.pkl"), "wb"
        ) as f:
            pickle.dump(encoded_all_intents, f)

        return {"file": file_name}
    
    
    # load model functions
    def load( self, 
        file_name: Text,
        model_dir: Text,
             ):
        # type: (...) -> EmbeddingIntentClassifier

        if model_dir and file_name:
            checkpoint = os.path.join(model_dir, file_name+ ".ckpt")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + '.meta')

                saver.restore(sess, checkpoint)

                a_in = tf.get_collection('message_placeholder')[0]
                b_in = tf.get_collection('intent_placeholder')[0]

                sim_op = tf.get_collection('similarity_op')[0]

                word_embed = tf.get_collection('word_embed')[0]
                intent_embed = tf.get_collection('intent_embed')[0]

            with io.open(os.path.join(
                    model_dir,
                    file_name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            return {
                    "inv_intent_dict":inv_intent_dict,
                    "encoded_all_intents":encoded_all_intents,
                    "session":sess,
                    "graph":graph,
                    "a_in":a_in,
                    "b_in":b_in,
                    "sim_op":sim_op,
                    "word_embed":word_embed,
                    "intent_embed":intent_embed
            }

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return {}
        
    
    
    
    # predict functions
    def _calculate_message_sim(self, 
        X: np.ndarray, all_Y: np.ndarray,
        session: "tf.Session",
        a_in: "tf.placeholder",
        b_in: "tf.placeholder",
        sim_op:"tf.Tensor",
    ) -> Tuple[np.ndarray, List[float]]:
        """Load tf graph and calculate message similarities"""

        message_sim = session.run(
            sim_op, feed_dict={a_in: X, b_in: all_Y}
        )
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        if self.defaults["similarity_type"]  == "cosine":
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.defaults["similarity_type"]  == "inner":
            # normalize result to [0, 1] with softmax
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return intent_ids, message_sim.tolist()

    def process(self, 
            message: List,
            session: "tf.Session",
            inv_intent_dict: Dict,
            encoded_all_intents: np.ndarray,
            a_in: "tf.placeholder",
            b_in: "tf.placeholder",
            sim_op:"tf.Tensor"
           ) -> None:
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            pass

        else:
            # get features (bag of words) for a message
            # noinspection PyPep8Naming
            X = message.reshape(1, -1) #message.get("text_features").reshape(1, -1)
            

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            # noinspection PyPep8Naming
            all_Y = self._create_all_Y(X.shape[0],encoded_all_intents)

            # load tf graph and session
            intent_ids, message_sim = self._calculate_message_sim(X, all_Y,session,a_in,b_in,sim_op)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {
                    "name": inv_intent_dict[intent_ids[0]],
                    "confidence": message_sim[0],
                }

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:self.INTENT_RANKING_LENGTH]
                intent_ranking = [
                    {"name": inv_intent_dict[intent_idx], "confidence": score}
                    for intent_idx, score in ranking
                ]

        return {"intent": intent, "intent_ranking": intent_ranking}
    
    
    
    
    
    
    
    
    # train model functions
    def _create_intent_dict(self, training_data: TrainDataClass) -> Dict[Text, int]:
        """Create intent dictionary"""

        distinct_intents = set(
            [example.intent for example in training_data.data]
        )
        return {intent: idx for idx, intent in enumerate(sorted(distinct_intents))}

    
    def _create_intent_token_dict(self, 
        intents: List[Text], intent_split_symbol: Text
    ) -> Dict[Text, int]:
        """Create intent token dictionary"""

        distinct_tokens = set(
            [token for intent in intents for token in intent.split(intent_split_symbol)]
        )
        return {token: idx for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict: Dict[Text, int]) -> np.ndarray:
        """Create matrix with intents encoded in rows as bag of words.

        If intent_tokenization_flag is off, returns identity matrix.
        """

        if self.defaults["intent_tokenization_flag"]:
            intent_token_dict = self._create_intent_token_dict(
                list(intent_dict.keys()), self.defaults["intent_split_symbol"]
            )

            encoded_all_intents = np.zeros((len(intent_dict), len(intent_token_dict)))
            for key, idx in intent_dict.items():
                for t in key.split(self.defaults["intent_split_symbol"]):
                    encoded_all_intents[idx, intent_token_dict[t]] = 1

            return encoded_all_intents
        else:
            return np.eye(len(intent_dict))
    
    
    # noinspection PyPep8Naming
    def _prepare_data_for_training(self, 
        training_data: TrainDataClass, intent_dict: Dict[Text, int] ,encoded_all_intents: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""

        X = np.stack([e.intent_text_features for e in training_data.data])
        intents_for_X = np.array(
            [intent_dict[e.intent] for e in training_data.data]
        )

        Y = np.stack(
            [encoded_all_intents[intent_idx] for intent_idx in intents_for_X]
        )

        return X, Y, intents_for_X

    def _create_tf_embed_nn(self, 
        x_in: "tf.Tensor",
        is_training: "tf.Tensor",
        layer_sizes: List[int],
        name: Text,
    ) -> "tf.Tensor":
        """Create nn with hidden layers and name"""

        reg = tf.contrib.layers.l2_regularizer(self.defaults["C2"])
        x = x_in
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(
                inputs=x,
                units=layer_size,
                activation=tf.nn.relu,
                kernel_regularizer=reg,
                name="hidden_layer_{}_{}".format(name, i),
            )
            x = tf.layers.dropout(x, rate=self.defaults["droprate"], training=is_training)

        x = tf.layers.dense(
            inputs=x,
            units=self.defaults["embed_dim"],
            kernel_regularizer=reg,
            name="embed_layer_{}".format(name),
        )
        return x

    def _create_tf_embed(self, 
         a_in: "tf.Tensor", b_in: "tf.Tensor", is_training: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Create tf graph for training"""

        emb_a = self._create_tf_embed_nn(
            a_in, is_training, self.defaults["hidden_layers_sizes_a"], name="a"
        )
        emb_b = self._create_tf_embed_nn(
            b_in, is_training, self.defaults["hidden_layers_sizes_b"], name="b"
        )
        return emb_a, emb_b
    
    
    def _tf_sim(self, 
         a: "tf.Tensor", b: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Define similarity

        in two cases:
            sim: between embedded words and embedded intent labels
            sim_emb: between individual embedded intent labels only
        """

        if self.defaults["similarity_type"] == "cosine":
            # normalize embedding vectors for cosine similarity
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if self.defaults["similarity_type"] in {"cosine", "inner"}:
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)

            return sim, sim_emb

        else:
            raise ValueError(
                "Wrong similarity type {}, "
                "should be 'cosine' or 'inner'"
                "".format(defaults["similarity_type"])
            )
            
            
    def _tf_loss(self,  sim: "tf.Tensor", sim_emb: "tf.Tensor") -> "tf.Tensor":
        """Define loss"""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0.0, self.defaults['mu_pos'] - sim[:, 0])

        if self.defaults['use_max_sim_neg']:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
            loss += tf.maximum(0.0, self.defaults['mu_neg'] + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0.0, self.defaults['mu_neg'] + sim[:, 1:])
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between intent embeddings
        max_sim_emb = tf.maximum(0.0, tf.reduce_max(sim_emb, -1))
        loss += max_sim_emb * self.defaults['C_emb']

        # average the loss over the batch and add regularization losses
        loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()
        return loss
    

    def is_logging_disabled(self, ) -> bool:
        """Returns true, if log level is set to WARNING or ERROR, false otherwise."""
        log_level = "WARNING" #os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

        return log_level == "ERROR" or log_level == "WARNING"

    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489
        """

        if not isinstance(self.defaults['batch_size'], list):
            return int(self.defaults['batch_size'])

        if len(self.defaults['batch_size']) > 1:
            return int(
                self.defaults['batch_size'][0]
                + epoch * (self.defaults['batch_size'][1] - self.defaults['batch_size'][0]) / (self.defaults['epochs'] - 1)
            )
        else:
            return int(self.defaults['batch_size'][0])

        
        
    # training helpers:
    def _create_batch_b(self, 
        batch_pos_b: np.ndarray, intent_ids: np.ndarray, encoded_all_intents: np.ndarray
    ) -> np.ndarray:
        """Create batch of intents.

        Where the first is correct intent
        and the rest are wrong intents sampled randomly
        """

        batch_pos_b = batch_pos_b[:, np.newaxis, :]

        # sample negatives
        batch_neg_b = np.zeros(
            (batch_pos_b.shape[0], self.defaults['num_neg'], batch_pos_b.shape[-1])
        )
        for b in range(batch_pos_b.shape[0]):
            # create negative indexes out of possible ones
            # except for correct index of b
            negative_indexes = [
                i
                for i in range(encoded_all_intents.shape[0])
                if i != intent_ids[b]
            ]
            negs = np.random.choice(negative_indexes, size=self.defaults['num_neg'])

            batch_neg_b[b] = encoded_all_intents[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], 1)
    

    def _create_all_Y(self, size: int,encoded_all_intents: np.ndarray) -> np.ndarray:
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy
        """

        return np.stack([encoded_all_intents] * size)
    
    # noinspection PyPep8Naming
    def _output_training_stat(self, 
        X: np.ndarray, intents_for_X: np.ndarray, is_training: "tf.Tensor",a_in: "tf.placeholder",
        b_in: "tf.placeholder" ,sim_op:"tf.Tensor",encoded_all_intents: np.ndarray ,session: "tf.Session"
    ) -> np.ndarray:
        """Output training statistics"""

        n = self.defaults["evaluate_on_num_examples"]
        ids = np.random.permutation(len(X))[:n]
        all_Y = self._create_all_Y(X[ids].shape[0],encoded_all_intents)

        train_sim = session.run(
            sim_op,
            feed_dict={a_in: X[ids], b_in: all_Y, is_training: False},
        )

        train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X[ids])
        return train_acc
    
    def _train_tf(self, 
        X: np.ndarray,
        Y: np.ndarray,
        intents_for_X: np.ndarray,
        loss: "tf.Tensor",
        is_training: "tf.Tensor",
        train_op: "tf.Tensor",
        session: "tf.Session",
        encoded_all_intents: np.ndarray,
        a_in: "tf.placeholder",
        b_in: "tf.placeholder",
        sim_op:"tf.Tensor"
    ) -> None:
        """Train tf graph"""

        session.run(tf.global_variables_initializer())

        if self.defaults["evaluate_on_num_examples"]:
            logger.info(
                "Accuracy is updated every {} epochs"
                "".format(self.defaults["evaluate_every_num_epochs"])
            )
            pass

        pbar = tqdm(range(self.defaults["epochs"]), desc="Epochs", disable=self.is_logging_disabled())
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            indices = np.random.permutation(len(X))

            batch_size = self._linearly_increasing_batch_size(ep)
            batches_per_epoch = len(X) // batch_size + int(len(X) % batch_size > 0)

            ep_loss = 0
            for i in range(batches_per_epoch):
                end_idx = (i + 1) * batch_size
                start_idx = i * batch_size
                batch_a = X[indices[start_idx:end_idx]]
                batch_pos_b = Y[indices[start_idx:end_idx]]
                intents_for_b = intents_for_X[indices[start_idx:end_idx]]
                # add negatives
                batch_b = self._create_batch_b(batch_pos_b, intents_for_b,encoded_all_intents)

                sess_out = session.run(
                    {"loss": loss, "train_op": train_op},
                    feed_dict={
                        a_in: batch_a,
                        b_in: batch_b,
                        is_training: True,
                    },
                )
                ep_loss += sess_out.get("loss") / batches_per_epoch

            if self.defaults["evaluate_on_num_examples"]:
                if (
                    ep == 0
                    or (ep + 1) % self.defaults["evaluate_every_num_epochs"] == 0
                    or (ep + 1) == self.defaults["epochs"]
                ):
                    train_acc = self._output_training_stat(
                        X, intents_for_X, is_training ,a_in, b_in,sim_op,encoded_all_intents ,session
                    )
                    last_loss = ep_loss

                pbar.set_postfix(
                    {
                        "loss": "{:.3f}".format(ep_loss),
                        "acc": "{:.3f}".format(train_acc),
                    }
                )
            else:
                pbar.set_postfix({"loss": "{:.3f}".format(ep_loss)})

        if self.defaults["evaluate_on_num_examples"]:
            logger.info(
                "Finished training embedding classifier, "
                "loss={:.3f}, train accuracy={:.3f}"
                "".format(last_loss, train_acc)
            )
            pass
        
        
    def train_model(self, 
        training_data: TrainDataClass,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        intent_dict = self._create_intent_dict(training_data)
        if len(intent_dict) < 2:
            logger.error(
                "Can not train an intent classifier. "
                "Need at least 2 different classes. "
                "Skipping training of intent classifier."
            )
            return

        inv_intent_dict = {v: k for k, v in intent_dict.items()}
        encoded_all_intents = self._create_encoded_intents(intent_dict)

        # noinspection PyPep8Naming
        X, Y, intents_for_X = self._prepare_data_for_training(
            training_data, intent_dict, encoded_all_intents
        )

        # check if number of negatives is less than number of intents
        logger.debug(
            "Check if num_neg {} is smaller than "
            "number of intents {}, "
            "else set num_neg to the number of intents - 1"
            "".format(self.defaults['num_neg'], encoded_all_intents.shape[0])
        )
        
        
        self.defaults['num_neg'] = min(self.defaults['num_neg'], encoded_all_intents.shape[0] - 1)



        graph = tf.Graph()
        with graph.as_default():
            # set random seed
            np.random.seed(self.defaults['random_seed'])
            tf.set_random_seed(self.defaults['random_seed'])

            a_in = tf.placeholder(tf.float32, (None, X.shape[-1]), name="a")
            b_in = tf.placeholder(tf.float32, (None, None, Y.shape[-1]), name="b")

            is_training = tf.placeholder_with_default(False, shape=())

            (word_embed, intent_embed) = self._create_tf_embed(
                a_in, b_in, is_training
            )

            sim_op, sim_emb = self._tf_sim(word_embed, intent_embed)
            loss = self._tf_loss(sim_op, sim_emb)

            train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            session = tf.Session()

            self._train_tf(X, Y, intents_for_X, loss, is_training, train_op,session,encoded_all_intents ,a_in, b_in,sim_op)
            
            
            self.persist(file_name="intent_model",
            model_dir="intent_model",
            graph=graph,
            session=session,
            inv_intent_dict=inv_intent_dict,
            encoded_all_intents=encoded_all_intents,
            a_in=a_in,
            b_in=b_in,
            sim_op=sim_op,
            word_embed=word_embed,
            intent_embed=intent_embed,
           )
            
            
            self.model =  {
                    "inv_intent_dict":inv_intent_dict,
                    "encoded_all_intents":encoded_all_intents,
                    "session":session,
                    "graph":graph,
                    "a_in":a_in,
                    "b_in":b_in,
                    "sim_op":sim_op,
                    "word_embed":word_embed,
                    "intent_embed":intent_embed
            }
    