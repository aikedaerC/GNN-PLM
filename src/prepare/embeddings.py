import numpy as np
import torch
from torch_geometric.data import Data
from src.utils.functions.parse import tokenizer
from src.utils import log as logger
# from gensim.models.keyedvectors import Word2VecKeyedVectors
np.random.seed(42)  # Use any fixed seed value


class NodesEmbedding:
    def __init__(self, nodes_dim, embed_model, configs):
        self.configs = configs
        if self.configs.embed.embed_type in ["w2v", "ctg-former", "vulberta", "sam_vulberta"]:
            self.w2v_keyed_vectors = embed_model
            self.kv_size = self.w2v_keyed_vectors.vector_size
        elif self.configs.embed.embed_type in ["bert", "sam_bert"]:
            self.tokenizer_bert, self.bert_model, self.encode_input,_ = embed_model
            self.kv_size = self.bert_model.config.hidden_size

        self.nodes_dim = nodes_dim
        assert self.nodes_dim >= 0

        self.device = torch.device(configs.device)

        # Buffer for embeddings with padding
        self.target = torch.zeros(self.nodes_dim, self.kv_size + 1).float()

    def __call__(self, nodes):
        embedded_nodes = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        self.target[:nodes_tensor.size(0), :] = nodes_tensor

        return self.target

    def embed_nodes(self, nodes):
        embeddings = []

        for n_id, node in nodes.items():
            # Get node's code
            node_code = node.get_code()
            try:
                # Tokenize the code
                tokenized_code = tokenizer(node_code, True)
                if not tokenized_code:
                    # print(f"Dropped node {node}: tokenized code is empty.")
                    msg = f"Empty TOKENIZED from node CODE {node_code}"
                    logger.log_warning('embeddings', msg)
                    continue
            except Exception as e:
                # Log the exception message and any relevant details
                error_msg = f"Error tokenizing code from node {node_code}. Exception: {str(e)}"
                logger.log_warning('embeddings', error_msg)
                continue

                # *** UnicodeDecodeError: 'unicodeescape' codec can't decode bytes in position 10-12: truncated \xXX escape
                # "q2[0] == '\\xa"
            # ######################################################################################################### #
            # Get each token's learned embedding vector                                                                 #
            if self.configs.embed.embed_type in ["w2v", "ctg-former", "vulberta", "vulberta_sam"]:                                                                  # 
                vectorized_code = np.array(self.get_vectors(tokenized_code, node))  
                # The node's source embedding is the average of it's embedded tokens
                source_embedding = np.mean(vectorized_code, 0)                                                          #
            elif self.configs.embed.embed_type in ["bert", "sam_bert"]:
                try:                                                                                                        #
                    input_ids, attention_mask = self.encode_input(tokenized_code, self.tokenizer_bert)                      #
                    vectorized_code = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]   #
                    source_embedding = np.mean(vectorized_code.cpu().detach().numpy(), 0)
                except:
                    continue
            # ######################################################################################################### #
 
            # The node representation is the concatenation of label and source embeddings
            embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
            embeddings.append(embedding)
        # print(node.label, node.properties.properties.get("METHOD_FULL_NAME"))

        return np.array(embeddings)

    # ########################################################################################## #
    # fromTokenToVectors                                                                         #
    def get_vectors(self, tokenized_code, node):                                                 #
        vectors = []                                                                             #
                                                                                                 #
        for token in tokenized_code:                                                             #
            if token in self.w2v_keyed_vectors.vocab:                                            #
                vectors.append(self.w2v_keyed_vectors[token])                                    #
            else:                                                                                #
                # print(node.label, token, node.get_code(), tokenized_code)                      #
                vectors.append(np.zeros(self.kv_size))                                           #
                if node.label not in ["Identifier", "Literal", "MethodParameterIn",              #
                                      "MethodParameterOut"]:                                     #
                    msg = f"No vector for TOKEN {token} in {node.get_code()}."                   #
                    logger.log_warning('embeddings', msg)                                        #
                                                                                                 #
        return vectors                                                                           #
    # ########################################################################################## #



class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections, edge_attr = self.nodes_connectivity(nodes)
        edge_index = torch.tensor(connections, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        return edge_index, edge_attr

    def get_edge_features(self, edge, feature_dim=3):
        """
        Randomly initialize features for a given edge.
        Args:
            edge: The edge object (unused here since features are random).
            feature_dim: The dimension of the feature vector for each edge.
        Returns:
            A list of random features representing the edge.
        """
        # Generate a random feature vector of size `feature_dim`
        random_features = np.random.rand(feature_dim).tolist()
        
        return random_features
    
    # nodesToGraphConnectivity
    # TODO: there is still some problem
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]
        edge_attr = []

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for e_id, edge in node.edges.items():
                if edge.type != self.edge_type:
                    continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)
                    edge_attr.append(self.get_edge_features(edge))

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)
                    edge_attr.append(self.get_edge_features(edge))

        return coo, edge_attr


def nodes_to_input(nodes, target, func, nodes_dim, embed_model, edge_type, configs):
    nodes_embedding = NodesEmbedding(nodes_dim, embed_model, configs)
    graphs_embedding = GraphsEmbedding(edge_type)
    edge_index, edge_attr = graphs_embedding(nodes)
    label = torch.tensor([target]).float()

    return Data(x=nodes_embedding(nodes), edge_index=edge_index, edge_attr=edge_attr, y=label, func=func)
