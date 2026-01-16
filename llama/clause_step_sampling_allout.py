import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
import torch.nn.functional as F
import os.path
import pickle as pkl
import json
import numpy as np
from threading import Thread, Lock
import time
# import torchinfo
import inspect
import ast
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from collections import defaultdict , Counter
import math

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                args=(), kwargs={}, Verbose=None , daemon=True):
        Thread.__init__(self, group, target, name, args, kwargs , daemon=daemon)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
        
class LLM_Word_Level_Ensemble:

    def __init__(self, model_name ):
        # Initialize the model and the tokenizer.
        seed=26
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)#{"": accelerator.process_index}) ###NO DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_list = ['cuda:0' , 'cuda:1' , 'cuda:2' , 'cuda:3']
        self.model.eval()
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side="left"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id # updating model config)
        print(f'loading the model into two GPUs: {self.device_list}')

        seq_in_gpu_device = torch.nn.Sequential( self.model.model.embed_tokens ).to(self.device_list[0])
        self.model.model.rotary_emb.to(self.device_list[0])

        self.model.model.norm.to(self.device_list[-1])
        self.model.lm_head.to(self.device_list[-1])

        intro_memory = torch.cuda.max_memory_allocated(self.device_list[0])
        outro_memory = torch.cuda.max_memory_allocated(self.device_list[-1])
        layer_memory = self.get_layer_memory(self.model.model.layers[0])
        num_layers = len(self.model.model.layers)
        num_gpus = len(self.device_list)
        intro_to_layer_ratio = round(intro_memory/layer_memory)
        outro_to_layer_ratio = round(outro_memory/layer_memory)
        x = round( ( num_layers + intro_to_layer_ratio + outro_to_layer_ratio ) / num_gpus )# 2x+(x-1)+(x-1)=32 => x = 4x=34 => x=8
        self.partition_idx_list = [ x-intro_to_layer_ratio , 
                                    x-intro_to_layer_ratio + x,
                                    x-intro_to_layer_ratio + x + x] # 2x+(x-1)+(x-1)=32 => x = 4x=34 => x=8

        for index , layer in enumerate(self.model.model.layers):
            if index < self.partition_idx_list[0] :
                layer.to(self.device_list[0])
            elif index < self.partition_idx_list[1] :
                layer.to(self.device_list[1])
            elif index < self.partition_idx_list[2] :
                layer.to(self.device_list[2])
            else:
                layer.to(self.device_list[3])
        
        self.gpu_lock_list = [ Lock() for i in self.device_list]
        print('Model successfully loaded')

        self.sql_keywords = [
            "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET",
             "UNION", "WITH" , "INTERSECT" , "EXCEPT"
        ]

        
    def get_layer_memory(self , layer):
        # Calculate the memory occupied by the parameters of the layer
        param_size = 0
        for param in layer.parameters():
            param_size += param.numel() * param.element_size()  # numel() gives the number of elements in the tensor
                                                                # element_size() gives the size of each element in bytes

        return param_size

    def tokenize_sentences(self , sentences ):
        #tokenizes a batch of sentences
        #Input:
            #sentences: list of strings. length = batch_size
        #returns:
            #torch tensor representing the tokens in the sentences. shape(batch_size , maximum_length_of_a_sentence)
        self.last_inputs = self.tokenizer(sentences, return_tensors="pt" , padding=True)
        
        return self.last_inputs.input_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cache_position = None,
    ):
        for gpu_index , lock in enumerate(self.gpu_lock_list):  
            with lock:
                if gpu_index==0:
                    #Operations dedicated to the first gpu. It is the operations in the model that come before the main layers.
                    output_attentions=False
                    output_hidden_states = False
                    use_cache=True
                    return_dict=True

                    if attention_mask is None:
                        attention_mask = (input_ids != self.model.config.pad_token_id).type(torch.int64)

                    if inputs_embeds is None:
                        inputs_embeds = self.model.model.embed_tokens(input_ids)


                    if cache_position is None:
                        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

                        # Generate a cumulative position index ignoring padding tokens
                        # cumulative_positions = torch.cumsum(attention_mask, dim=-1) #- 1
                        # cache_position = cumulative_positions + past_seen_tokens
                        # print('cumultive cache_position: ' , cache_position)
                        cache_position = torch.arange(
                            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                        )
                    # print('cache_position.size(): ' , cache_position.size())
                    # print('cache_position1.size(): ' , cache_position1.size())
                    #This line is for test
                    # if position_ids is None:
                    #     position_ids = cache_position1.unsqueeze(0)

                    if position_ids is None:
                        position_ids = cache_position.unsqueeze(0)
                    # print('position_ids.size(): ' , position_ids.size())

                    #copied from _prepare_4d_causal_attention_mask_with_cache_position and modified
                    # sequence_length = inputs_embeds.size(1)
                    # target_length = attention_mask.size(-1)
                    # dtype = inputs_embeds.dtype
                    # device = inputs_embeds.device
                    # min_dtype = torch.finfo(dtype).min
                    # causal_mask = torch.full(
                    #     (sequence_length, target_length), fill_value=min_dtype, dtype=dtype , device=device
                    # )
                    # if sequence_length != 1:
                    #     causal_mask = torch.triu(causal_mask, diagonal=1)
                    # causal_mask = causal_mask[None, :, :].expand(attention_mask.size(0), -1, -1).clone()
                    # causal_mask *= torch.arange(target_length, device=device) > cache_position[:,:,None]
                    # causal_mask = causal_mask[ : , None , : , : ]
                    # if attention_mask is not None:
                    #     causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    #     mask_length = attention_mask.shape[ -1 ]
                    #     padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    #     padding_mask = padding_mask == 0
                    #     causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    #         padding_mask, min_dtype
                    #     )
                    #End of the code block taken from modeling_llama.py and modified
                    #The above block of code is a replacement for the below line of code
                    causal_mask = self.model.model._update_causal_mask(
                        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions )

                    # causal_mask = self.model.model._update_causal_mask(
                    #     causal_mask, inputs_embeds, cache_position, past_key_values, output_attentions )
                    # print('min_dtype: ' , min_dtype)
                    # print(causal_mask1 - causal_mask)
                    # print(f'attention_mask: {type(attention_mask)} {attention_mask.size()}')
                    # print(f'inputs_embeds: {type(inputs_embeds)} {inputs_embeds.size()}')
                    # print(f'cache_position: {type(cache_position)} {cache_position.size()}')
                    # print(f'past_key_values: {type(past_key_values)}')
                    # print(f'output_attentions: {type(output_attentions)}')
                    # print(f'type(causal_mask): {type(causal_mask)}')
                    # print(f'causal_mask.size(): {causal_mask.size()}')
                    hidden_states = inputs_embeds

                    # create position embeddings to be shared across the decoder layers
                    position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

                    next_decoder_cache = None

                    # decoder layers
                    for idx , decoder_layer in enumerate(self.model.model.layers[:self.partition_idx_list[0]]):
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        hidden_states = layer_outputs[0].clone()
                        del layer_outputs; #torch.cuda.empty_cache()
                    device_recently_changed = True

                elif gpu_index<len(self.gpu_lock_list)-1:
                    #operations for the middle gpus working only on the middle layers
                    for idx , decoder_layer in enumerate(
                        self.model.model.layers[self.partition_idx_list[gpu_index-1]:self.partition_idx_list[gpu_index]]):
                        if device_recently_changed==True:
                            hidden_states = hidden_states.to( self.device_list[gpu_index] )
                            if causal_mask is not None:
                                causal_mask = causal_mask.to( self.device_list[gpu_index] )
                            # del layer_outputs; torch.cuda.empty_cache()
                            position_ids = position_ids.to( self.device_list[gpu_index] )
                            position_embeddings = None
                            device_recently_changed=False
                
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        hidden_states = layer_outputs[0].clone()
                        del layer_outputs; #torch.cuda.empty_cache()
                    device_recently_changed = True
                else: # gpu_index == len(self.gpu_lock_list)-1
                    #operations for the last GPU. the outro of the model
                    for idx , decoder_layer in enumerate(self.model.model.layers[self.partition_idx_list[-1]:]):
                        if device_recently_changed==True:
                            hidden_states = hidden_states.to(self.device_list[gpu_index])
                            if causal_mask is not None:
                                causal_mask = causal_mask.to(self.device_list[gpu_index])
                            position_ids = position_ids.to(self.device_list[gpu_index])
                            position_embeddings = None
                            device_recently_changed=False
                        
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        hidden_states = layer_outputs[0]
                        # del layer_outputs; #torch.cuda.empty_cache()
                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    hidden_states = self.model.model.norm(hidden_states)
                    
                    logits = self.model.lm_head(hidden_states[:, -1, :])

                    next_cache = next_decoder_cache if use_cache else None
        return logits , next_cache
    
    def get_predictions(self, inputs , past_key_values=None):
        #performs the prediction on the tokenized sentences.
        #input:
            #torch tensor of shape(batch_size , length_of_sentences)
        #output: prediction vectors of shape: (batch_size , length_of_sent , vocab_size)
        with torch.no_grad():
            return self.forward( inputs , past_key_values=past_key_values )

    def replace_token_with_eos(self, token_id , indice_vector , prob_vector , change_prob=False):
        mask = (indice_vector!=token_id).type(torch.int64).to(indice_vector.device)
        unmask = ( (indice_vector==token_id).type(torch.int64) ).to(indice_vector.device)
        if change_prob:
            prob_vector = (prob_vector * mask) + unmask * prob_vector[:,-1:,:]
        indice_vector = (indice_vector * mask) + unmask * self.tokenizer.eos_token_id
        return prob_vector,indice_vector
    
    def get_top_words_id_prob_from_score_vector(self , word_scores, num_returned_words=1):
        #Gets the top singular word from a probability vecotr randomly based on its probability in the given vector and returns them.
        #Input:
            #word_scores: (batch_size , vocab_size , num_beam)
        #Retrun:
            #torch tensor on the same device as before (number_of_processing_prompts , num_returned_words , num_beam)
        topk_candidates_indexes = torch.topk(
            word_scores, k= num_returned_words , dim=1).indices
        # print('word_scores: ' , word_scores)
        word_probabilities = torch.nn.functional.softmax(
            word_scores, dim=1)
        word_probabilities = word_probabilities * ( word_scores != 0 )

        batch_indice = self.make_batch_indice_tensor( word_scores.size() , num_returned_words) #(batch_size , num_returned_words , num_beam)
        beam_indice = self.make_batch_indice_tensor( ( word_scores.size(-1), word_scores.size(0) ) , num_returned_words).transpose(0,2) #(batch_size , num_returned_words , num_beam)
        topk_candidates_prob = word_probabilities[ batch_indice , topk_candidates_indexes , beam_indice]#(batch_size , top-k , num_beam)
        
        # prob_vector , indice_vector = self.a( self.tokenizer.bos_token_id , topk_candidates_indexes , topk_candidates_prob)
        prob_vector , indice_vector = self.replace_token_with_eos( 5515 , topk_candidates_indexes , topk_candidates_prob)
        prob_vector , indice_vector = self.replace_token_with_eos( 29905 , indice_vector , prob_vector)

        return indice_vector.to(torch.int64), prob_vector
    
    def make_batch_indice_tensor(self , batch_size, expansion_size):
        #This function creates indice for each element in the batch in a tensor so that it can be used to select some tokens or parts of each element in a batch
        #Input:
            #batch_size: integer. Size of the batch
            #expansion_size: Integer. Number of enries that we want to get from each batch so that we should expand the tensor with this amount
        #Return:
            #A torch tensor with the shape(batch_size , expansion_size) each element of X_{ij} = i
        if isinstance(batch_size, tuple):
            batch_indice_tensor = torch.tensor( list(range( batch_size[0] )) ) # shape (batch_size,)
            batch_indice_tensor = torch.reshape(batch_indice_tensor , (batch_size[0] , 1 , 1)) #shape(batch_size , 1 , 1)
            batch_indice_tensor = batch_indice_tensor.expand( batch_size[0] , expansion_size , batch_size[-1]) #shape ( batch_size , top_k )
        else:
            batch_indice_tensor = torch.tensor( list(range( batch_size )) ) # shape (batch_size,)
            batch_indice_tensor = torch.reshape(batch_indice_tensor , (batch_size , 1)) #shape(batch_size , 1)
            batch_indice_tensor = batch_indice_tensor.expand( batch_size , expansion_size ) #shape ( batch_size , top_k )
        return batch_indice_tensor
    
    def check_if_start_of_clause(self, sentences):
        #inputs:
            #sentences: torch tensor of size(batch size, input_len).
        #return:
            #torch tensor containing true or false stating if the sentecne has a reserved word or not
        decoded_sentences = self.tokenizer.batch_decode( sentences , skip_special_tokens=False )
        output_list = []
        for sent in decoded_sentences:
            found_keyword = False
            for keyword in self.sql_keywords:
                if sent.upper()[-(len(keyword)):] == keyword:
                    found_keyword = True
                    # print('found keyword')
                    break
            output_list.append(found_keyword)
        return torch.tensor( output_list ).to(sentences.device)

    def get_next_word_score(self, sentences, just_performed_ensemble, current_clause_length, past_key_values=None ,top_k=5):#operation_on_a_batch
        # Getting the word probabilities for the next word. It will have the probability for the top-k words, and it will zero out the rest. If the top_k argument is set to -1 then it will return the whole probability distribution.
        #Input:
            #sentences: torch tensor of size(batch size, input_len).
            #top_k: integer. Number of words with the highest probability that we want to have their probability back in the returning vecotr
        #Retrun:
            #torch tensor. The vector containing the probability of top_k words for the next word. The rest of the words are zeroed out. shape(batch_size , vocab_size)
        
        batch_size = len(sentences)
        # print('batch_size: ' , batch_size)
        next_token_candidates_tensor = torch.zeros( (batch_size , self.vocab_size) ).to(self.device_list[-1])
        last_tokens = sentences[:,-1]
        
        non_active_texts_index = torch.nonzero( last_tokens==self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')

        next_token_candidates_tensor[non_active_texts_index,:] = 0
        next_token_candidates_tensor[non_active_texts_index, self.tokenizer.eos_token_id ] = 1
        
        if just_performed_ensemble == False:
            current_clause_length +=1
            is_waiting = self.check_if_start_of_clause(sentences)
            # print(is_waiting)
            # print(True in is_waiting)
            is_active = torch.logical_and( torch.logical_not(is_waiting) , last_tokens!=self.tokenizer.eos_token_id )
            if False not in is_active:
                current_clause_length = 1
            # print('current_clause_length: ' , current_clause_length)
            active_texts_index = torch.nonzero( is_active ).reshape( (-1,) ).to('cpu')
            reached_end_of_clause = torch.nonzero( is_waiting ).reshape( (-1,) ).to('cpu')
            next_token_candidates_tensor[reached_end_of_clause,:] = 0
            next_token_candidates_tensor[reached_end_of_clause, self.tokenizer.pad_token_id ] = 1
            # waiting_texts_index = torch.nonzero( last_tokens==self.tokenizer.pad_token_id ).reshape( (-1,) ).to('cpu')
            # print( 'waiting_texts_index: ' , waiting_texts_index )
            # next_token_candidates_tensor[waiting_texts_index,:] = 0
            # next_token_candidates_tensor[waiting_texts_index, self.tokenizer.pad_token_id ] = 1
        else:
            current_clause_length = 1
            active_texts_index = torch.nonzero( last_tokens!=self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')
            just_performed_ensemble = False

        if active_texts_index.size(0)>0:
            logits , past_key_values = self.get_predictions(sentences[active_texts_index,:] , 
                                                            past_key_values= past_key_values) #shape(batch_size , vocab_size)
            next_token_candidates_tensor[active_texts_index,:] = logits
        
        # print('active_texts_index: ' , active_texts_index)
        # print('non_active_texts_index: ' , non_active_texts_index)
        # print('reached_end_of_clause: ' , reached_end_of_clause)
        
        #If we want to apply masking, the following block of code should be uncommented
        # topk_candidates_indexes = torch.topk(
        #     next_token_candidates_tensor, k=top_k , dim=1 ).indices #shape( batch_size , top_k , num_beam)

        # batch_indice_tensor = self.make_batch_indice_tensor(batch_size , top_k)

        # masking_matrix = torch.zeros( ( next_token_candidates_tensor.size() ) ).to(self.device_list[-1])
        # masking_matrix[ batch_indice_tensor , topk_candidates_indexes ] = 1
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_tensor_score = next_token_candidates_tensor #* masking_matrix
        
        # del sentences; torch.cuda.empty_cache()
        return topk_candidates_tensor_score , past_key_values, just_performed_ensemble, current_clause_length #first output shape (batch_size , vocab_size)
        

    # import os.path
    # import pickle as pkl
    # import json
    # import numpy as np
    def load_prompts_from_datasets(self , directory , dataset_list , starting_index=0, ending_index=-1):
        #getting the prompts from the datasets keeping them in a array accordingly, storing the list of prompts in a file for later use. Getting a numpy array ready for LLM for inference in self.all_prompts_in_1D.
        #input:
            #dataset_list: list of string. a list of addresses for the json file datasets.
            #directory: address of the directory that datasets are in. It ends with '/'
            #starting_index: The starting index of prompts that we want to process
            #ending_index: The ending index of prompts that we want to process
        #return:
            #length of the list containing all the prompts
        datasets_prompts_list = []
        for dataset in dataset_list:
            prompt_list = []
            potential_cache_file_name = './ensemble_cache/' + dataset+'.pkl'
            if os.path.isfile(potential_cache_file_name):
                with open(potential_cache_file_name, 'rb') as f:
                    prompt_list = pkl.load(f)
            else:
                dataset_path = directory + dataset# + '/' + 'questions.json'
                with open( dataset_path , 'r') as f:
                    dataset_file_byte = f.read()
                    dataset_json_format = json.loads(dataset_file_byte)
                question_units = dataset_json_format['questions']
                for question_unit in question_units:
                    prompt_list.append(question_unit['prompt'].strip())
                with open(potential_cache_file_name, 'wb') as f:
                    pkl.dump( prompt_list , f )
            if ending_index==-1:
                datasets_prompts_list.append( prompt_list[ starting_index : ] )
            else:
                datasets_prompts_list.append( prompt_list[ starting_index : ending_index ] )
        datasets_prompts_array = np.array(datasets_prompts_list ) #shape (number_of_datasset , number_of_prompts)
        self.all_prompts_in_1D = datasets_prompts_array.flatten('F').tolist()
        
        self.number_of_components = datasets_prompts_array.shape[0]
        self.number_of_prompts = datasets_prompts_array.shape[1]
        self.vocab_size = len(self.tokenizer)
        return len(self.all_prompts_in_1D)
    
    # from collections import defaultdict
    def group_tokenized_responses(self, tokenized_responses, inputs_log_prob):
        # Dictionary to group indices of identical tokenized responses
        groups = defaultdict(list)

        # Iterate over the tokenized responses with their indices
        for i, tokens in enumerate(tokenized_responses):
            # Convert tokens to a tuple (hashable) to use as a dictionary key
            token_bag_with_counts = tuple(sorted(Counter(tokens).items()))
            groups[token_bag_with_counts].append(i)

        # print('groups.keys(): ', groups.keys())
        # print('groups: ' , groups)
        # Convert the grouped dictionary values into a list of groups
        grouped_indices = list(groups.values())

        # Calculate the average log probabilities for each group and assign them
        for group in grouped_indices:
            # Extract log probabilities for the current group
            group_probs = [
                inputs_log_prob[int(idx % self.number_of_components), 0, int(idx / self.number_of_components)].item()
                for idx in group
            ]

            # Calculate the average probability for the group
            avg_prob = np.mean(group_probs)

            # Assign the average probability back to each index in the group
            for idx in group:
                inputs_log_prob[int(idx % self.number_of_components), 0, int(idx / self.number_of_components)] = avg_prob

        return grouped_indices, inputs_log_prob

    #calculating the similarity score of the two tokenized sentences.
    def similarity_meassure(self, hypothesis_tokens , reference_tokens, in_table_keywords):

        # hypothesis_tokens = [token for token in hypothesis_tokens if token.upper() not in self.sql_keywords]
        # reference_tokens = [token for token in reference_tokens if token.upper() not in self.sql_keywords]

        hypothesis_counts = Counter(hypothesis_tokens)
        reference_counts = Counter(reference_tokens)
        clipped_counts = dict()

        for token in hypothesis_tokens:
            if token not in reference_counts.keys():
                reference_counts[token] = 0
            if token in in_table_keywords:
                clipped_counts[token] = min(hypothesis_counts[token], reference_counts[token])
            else:
                clipped_counts[token] = min(hypothesis_counts[token], reference_counts[token])/2
        
        total_clipped = sum(clipped_counts.values())

        similarity = (total_clipped*2+1) / ( len(hypothesis_tokens) + len(reference_tokens) + 1 )
        
        return math.log(similarity)

    def ensemble_bleu(self, inputs_ids , inputs_log_prob , starting_batch_input_len , extra_added_paddings, batch_text ):
        #function for performing ensemble using the bleu metric between the candidate sequences.
        #input:
            #inputs_ids: torch tensor representing the prompt tokens per component with shape (batch_size , input_len , num_beam)
            #inputs_log_prob: torch tensor representing the probability of each input_ids with the shape(batch_size , 1 , num_beam)
            #starting_batch_input_len: The length of the batch before any predictions.
            #extra_added_paddings: torch tensor with size (batch_size , 1 , num_beam) recording the token index of the last token of the prompt (where the answer starts)
            #batch_text: list of string with len()=number_components. Having the prompts for each components.
        #return:
            #ensembled_inputs_ids: torch tensor with size (batch_size , input_len , num_beam)
            #ensembled_inputs_log_prob: torch tensor with size (batch_size , 1 , num_beam)
        # print('extra_added_paddings before ensemble: ' , extra_added_paddings)
        batch_size = inputs_ids.size( dim=0 )
        num_beam = inputs_ids.size( dim=-1 )
        ensembled_inputs_ids = inputs_ids.clone()
        ensembled_inputs_log_prob = inputs_log_prob.clone()
        for i in range ( 0, batch_size ,self.number_of_components ): #number_of_processing_prompts
            table_creation_part_prompt = batch_text[i].split('Given the following database schema:')[-1].split('Answer the following')[0]
            # print('table_creation_part_prompt: ', table_creation_part_prompt)
            in_table_keywords = word_tokenize( table_creation_part_prompt )
            # print('in_table_keywords: ' , in_table_keywords)
            tokenized_responses = []
            decoded_text_list = []
            last_component = i + self.number_of_components
            #Tokenizing the candidate sequences
            for j in range(num_beam):
                components_token_list = []
                for component in range(i,last_component,1):
                    components_token_list.append( inputs_ids[ component ,
                     int( starting_batch_input_len+extra_added_paddings[component,0,j].item() ): , j ] )
                decoded_text_list.extend( self.tokenizer.batch_decode( components_token_list ,
                                                skip_special_tokens=True ) )
            for text in decoded_text_list: #number of candidates we have for each question
                # print(text)
                tokenized_responses.append( word_tokenize( text.replace('.' , ' ') ) )
            selection_score_list = [] #This scoring is used to select the best candidates. It uses the length penalty to calculate the scores
            updating_score_list = [] #This scoring is used to update the score of selected candidates. It does not use the length penalty to calculate the scores

            #Finding the identical candidates, take the average of their probability, and only keep one of them with the average probability assigned to it.
            grouped_indices, inputs_log_prob = self.group_tokenized_responses( tokenized_responses, inputs_log_prob)
            gen_text_len = self.input_ids_to_gen_text_len(inputs_ids , starting_batch_input_len , extra_added_paddings) #shape(batch_size , 1 , num_candidate_beams)
            tmp_input_log_prob = inputs_log_prob/(gen_text_len**0.1)

            #Calculating the bleu metrics for each candidate
            for j in range( len( tokenized_responses ) ): #[tok_component1_beam1, tok_component2_beam1, tok_component3_beam1, ..., tok_component1_beam2, tok_component2_beam2 , ...]
                temp_tokenized_responses = tokenized_responses.copy()
                tokenized_response = temp_tokenized_responses.pop(j)
                selection_score = 0
                updating_score = 0
                for index , other_response in enumerate(temp_tokenized_responses):
                    if index>=j:
                        index+=1
                    # other_response_prob = torch.exp( inputs_log_prob[ int((index%self.number_of_components)+i) , 0,  int(index/self.number_of_components) ] )
                    selection_score_with_other_response = tmp_input_log_prob[ int((index%self.number_of_components)+i) , 0,  int(index/self.number_of_components) ] + self.similarity_meassure( tokenized_response, other_response, in_table_keywords )
                    updating_score_with_other_response = inputs_log_prob[ int((index%self.number_of_components)+i) , 0,  int(index/self.number_of_components) ] + self.similarity_meassure( tokenized_response, other_response, in_table_keywords )
                    # score += self.similarity_meassure( tokenized_response, other_response ) * other_response_prob
                    if selection_score == 0:
                        selection_score = selection_score_with_other_response
                        updating_score = updating_score_with_other_response
                    else:
                        alpha = max(selection_score_with_other_response , selection_score)
                        beta = min(selection_score_with_other_response , selection_score)
                        selection_score = alpha + torch.log1p(torch.exp(beta-alpha))

                        alpha = max(updating_score_with_other_response , updating_score)
                        beta = min(updating_score_with_other_response , updating_score)
                        updating_score = alpha + torch.log1p(torch.exp(beta-alpha))
                # log_score = torch.log( score )#/ len(temp_tokenized_responses) )
                selection_log_score = (selection_score + tmp_input_log_prob[ int((j%self.number_of_components)+i) , 0,  int(j/self.number_of_components) ] )#/2
                updating_log_score = (updating_score + inputs_log_prob[ int((j%self.number_of_components)+i) , 0,  int(j/self.number_of_components) ] )#/2
                # print(f'toknes:{tokenized_response} point:{score}')
                selection_score_list.append(selection_log_score)
                updating_score_list.append(updating_log_score)
            

            for group in grouped_indices:
                is_first_item = True
                for index in group:
                    if is_first_item ==False:
                        selection_score_list[index] = -100000
                    else:
                        is_first_item = False
            # print('\nAfter grouping:\n')
            # for j in range( len( tokenized_responses ) ): 
            #     print(f'toknes:{tokenized_responses[j]} point:{selection_score_list[j]}')

            selected_candidate_list = [] #containing tuples like (component_index , beam_index)
            selected_candidate_ex_add_tok = [] #It shows the number of extra added tokens for each of the selected candidates
            #Finding the sequence with the highest bleu score.
            tmp_score_list = selection_score_list.copy()
            for beam in range(num_beam):
                max_bleu_score_value = max( tmp_score_list )
                max_index_bleu_score = selection_score_list.index(max_bleu_score_value)
                tmp_score_list[ max_index_bleu_score ] = -100000
                selected_candidate_list.append( ( int(max_index_bleu_score%self.number_of_components) , int(max_index_bleu_score/self.number_of_components) ) )
                selected_candidate_ex_add_tok.append( extra_added_paddings[selected_candidate_list[beam][0]+i , 0 , selected_candidate_list[beam][1]].item() )
            # print('selected_candidate_list: ' , selected_candidate_list)
            for component_index in range(self.number_of_components):
                for beam_index in range(len(selected_candidate_list)):
                    start_of_selected_gen_text = int(starting_batch_input_len + selected_candidate_ex_add_tok[beam_index])
                    size_of_input_prompt = int(starting_batch_input_len + extra_added_paddings[i+component_index,0,beam_index].item())
                    index_diff = size_of_input_prompt - start_of_selected_gen_text
                    extra_added_paddings[i+component_index,0,beam_index] = selected_candidate_ex_add_tok[beam_index]
                    if index_diff>0:
                        padding_tensor = torch.full( (index_diff,), self.tokenizer.pad_token_id, dtype=ensembled_inputs_ids.dtype, device=ensembled_inputs_ids.device)
                        ensembled_inputs_ids[i+component_index ,: , beam_index ]  = torch.cat( (ensembled_inputs_ids[i+component_index , index_diff: , beam_index ] , padding_tensor) )
                    elif index_diff<0:
                        padding_tensor = torch.full( (-index_diff,), self.tokenizer.pad_token_id, dtype=ensembled_inputs_ids.dtype, device=ensembled_inputs_ids.device)
                        ensembled_inputs_ids[i+component_index ,: , beam_index ]  = torch.cat( (padding_tensor , ensembled_inputs_ids[i+component_index , :index_diff , beam_index ]) )
                    ensembled_inputs_ids[i+component_index , 
                    start_of_selected_gen_text: , beam_index ] = inputs_ids[selected_candidate_list[beam_index][0]+i,
                                                                    start_of_selected_gen_text: , selected_candidate_list[beam_index][1] ]
                    # print(f'orig log prob: {inputs_log_prob[selected_candidate_list[beam_index][0]+i, 0, selected_candidate_list[beam_index][1]]}')
                    # print(f'ensemble prob: {score_list[self.number_of_components * selected_candidate_list[beam_index][1] + selected_candidate_list[beam_index][0]]}')
                    # ensembled_inputs_log_prob[i+component_index , 0 , beam_index] = ( inputs_log_prob[selected_candidate_list[beam_index][0]+i,
                    #                                                                 0, selected_candidate_list[beam_index][1]] + score_list[self.number_of_components * selected_candidate_list[beam_index][1] + selected_candidate_list[beam_index][0]] )/2
                    ensembled_inputs_log_prob[i+component_index , 0 , beam_index] = updating_score_list[self.number_of_components * selected_candidate_list[beam_index][1] + selected_candidate_list[beam_index][0]]
        # print('extra_added_paddings after ensemble: ' , extra_added_paddings)
        return ensembled_inputs_ids , ensembled_inputs_log_prob

    
        
    # import torch.nn.functional as F
    def output_handeling(self, batch_output ,  input_ids , input_probs , starting_batch_input_len , extra_added_paddings,  num_return_input=1 , top_k=1):
        # This function takes predicted output of the model and extract top tokens and add those top tokens to the input. It also update the probability of each input
        # input:
        #     batch_output: the output of operation_on_a_batch returns. a torch tensor shape (batch_size , vocab_size , num_beams ) it is on self.other_device
            # input_ids: The input ids of the model that predicted the batch_output (batch_size , input_length , num_beams) 
            # input_probs: contains the probability of each input sequence (batch_size , 1 , num_beam)
        # return:
        #     new_input_ids that now the newly predicted tokens are added to them on self.device_list[0]
            # new_input_probs: probability of each sequence in input_ids. shape (batch_size , 1 , num_candidate_beams ) num_candidate_beams = num_beam*num_return_input
        batch_size = batch_output.size(dim=0)
        num_beam = batch_output.size(dim=-1)
        number_of_processing_prompts = int( batch_size / self.number_of_components )
        
        new_words_vector, new_words_prob = self.get_top_words_id_prob_from_score_vector( batch_output , num_returned_words=num_return_input) #shape (batch_size , num_returned_words , num_beam)
        # print(new_words_vector , '\n')
        new_words_vector= new_words_vector.reshape(-1, 1, num_return_input*num_beam).to(self.device_list[0])  # shape(batch_size , 1 , num_beam*num_return_input) batch_size = number_of_processing_prompts*number_of_components

        input_ids = input_ids.repeat( 1,1,num_return_input) #shape (batch_size , input_len , num_beam*num_return_input)
        extra_added_paddings = extra_added_paddings.repeat( 1,1,num_return_input)#shape (batch_size , 1 , num_beam*num_return_input)
        
        #Masking for putting eos token if the last token is eos regardless of the predicted token
        token_mask = (input_ids[:,-1:,:]!=self.tokenizer.eos_token_id)
        token_unmask = (input_ids[:,-1:,:]==self.tokenizer.eos_token_id) * self.tokenizer.eos_token_id
        new_words_vector = new_words_vector * token_mask + token_unmask
        
        new_words_prob = new_words_prob.reshape(-1,1,num_return_input*num_beam).to(self.device_list[0])  # shape(batch_size , 1 , num_beam*num_return_input)

        #masking to avoid adding tokens with zero probability
        token_mask = (new_words_vector != self.tokenizer.pad_token_id)
        token_umask = (new_words_vector == self.tokenizer.pad_token_id) * 1
        new_words_prob = new_words_prob * token_mask + token_umask

        new_input_ids = torch.cat( ( input_ids , new_words_vector ) , dim=1 ) #shape (batch_size , input_length+1 , num_beam*num_return_input)

        input_probs = input_probs.repeat(1,1,num_return_input) #shape (batch_size , 1 , num_beam*num_return_input)

        eos_masking = torch.eq( self.input_ids_to_gen_text_len( new_input_ids , starting_batch_input_len , extra_added_paddings) , (new_input_ids.size(1)-starting_batch_input_len-extra_added_paddings).to(self.device_list[0]) )
        # print(bad_candidates_masking)
        new_input_probs = input_probs + torch.log(new_words_prob) * eos_masking #shape (batch_size , 1 , num_beam*num_return_input)

        # del input_ids; del input_probs; del batch_output; #torch.cuda.empty_cache()
        
        return new_input_ids , new_input_probs , extra_added_paddings# (batch_size , input_len , num_candidate_beams ) and (batch_size , 1 , num_candidate_beams ) and (batch_size , 1 , num_candidate_beams ) num_candidate_beams = num_beam*num_return_input
            
    def print_predictions(self, inputs_ids , input_prob=None , starting_batch_input_len=0 ,extra_added_paddings=None ,  skip = False):
        #input_ids: (batch_size , input_len , num_beam)
        if input_prob is None:
            input_prob = torch.zeros( (inputs_ids.size(0) , 1 , inputs_ids.size(-1) ) )
        gen_text_list = []
        gen_text_len_list = []
        for i  in range( 0 , inputs_ids.size(0) , 1):#self.number_of_components ):
            # print(f'Index in batch: {int( (i+1)/self.number_of_components )}')
            beams_batch = []
            for beam in range(inputs_ids.size(-1)):
                start_of_beam = starting_batch_input_len + extra_added_paddings[i,0,beam] #the start of the generated text for the beam.
                start_of_beam = int(start_of_beam.item())
                beams_batch.append(inputs_ids[i,start_of_beam:,beam])
            input_text_list = self.tokenizer.batch_decode( beams_batch ,
                                                          skip_special_tokens=skip )
            gen_text_list.extend( input_text_list )
            start_of_first_beam = starting_batch_input_len + extra_added_paddings[i,0,0] #the start of the generated text for the first beam.
            gen_text_len_list.append( inputs_ids.size(1) - start_of_first_beam )
            print('number of the component: ', i)
            for j in range( len(input_text_list) ):
                print(f'Beam number {j}, prob={input_prob[i,0,j]} --> {input_text_list[j]} --> {inputs_ids[i,-1,j]}')
        return gen_text_list, gen_text_len_list


    def input_ids_to_gen_text_len(self , input_ids , starting_batch_input_len , extra_added_paddings):
    #gent_text_len = length of the generated text so far shape(batch_size , 1 , num_candidate_beams)
        #Input:
            #input_ids: shape(batch_size , input_len , num_candidate_beams)
            #starting_batch_input_len: int. index of where the generated text starts.
        output_shape = (input_ids.size(0) , 1 , input_ids.size(-1))
        cpu_input_ids = input_ids.to('cpu')
        input_ids_current_len = cpu_input_ids.size(1)
        end_idx_batch_input = torch.full(output_shape, input_ids_current_len)
        non_zero_vector = (cpu_input_ids == self.tokenizer.eos_token_id).nonzero()
        for i in range(non_zero_vector.size(0)):
            batch_num = non_zero_vector[i,0]
            beam_num = non_zero_vector[i,2]
            end_idx = non_zero_vector[i,1]
            if end_idx_batch_input[batch_num , 0 , beam_num] == input_ids_current_len:
                end_idx_batch_input[batch_num , 0 , beam_num]= end_idx+1
        end_idx_batch_input = end_idx_batch_input - extra_added_paddings #adding this line to address the extra added padding tokens to the begginning
        return end_idx_batch_input.add(-starting_batch_input_len).to(self.device_list[0])
        
        
    def shift_padding_to_start(self , tensor , extra_added_paddings):
        #This function brings the padding tokens that are at the end of a sequence to the begginning of the sequence
        #input:
            # tensor: with the shape (batch_size*num_beam , input_len )
            # extra_added_paddings: with the shape (batch_size , 1 , num_beam)
        #return:
            # result: tensor with the shape (batch_size*num_beam , input_len )
            # new_extra_added_paddings: with the shape (batch_size , 1 , num_beam)
        pad_token = self.tokenizer.pad_token_id
        # Get the last column (i.e., last token in each row)
        last_column = tensor[:, -1]
        
        # Identify rows where the last token is a padding token
        rows_with_padding = (last_column == pad_token).to('cpu') #shape (batch_size*num_beam ,)
        new_extra_added_paddings = extra_added_paddings + rows_with_padding.reshape(extra_added_paddings.size(0) , extra_added_paddings.size(-1) , 1 ).transpose(1,2)
        # Create a copy of the tensor to avoid in-place modification
        result = tensor.clone()
        
        # For rows with padding at the end, shift it to the start
        result[rows_with_padding] = torch.cat(
            (torch.full((rows_with_padding.sum(), 1), pad_token, dtype=tensor.dtype, device=tensor.device),
            tensor[rows_with_padding, :-1]), dim=1)
        
        return result , new_extra_added_paddings

    #from threading import Thread
    def pipeline_process(self, batch_size , log_file , num_beam=1 , top_k=1 , max_num_token=200 , start_index=0 , end_index=-1 , thread_number=0):
        # works on self.all_prompts_in_1D as data. devides the data into batch_sizes and performs the operation on a batch. It handels the output of each operation on a batch
        #inputs:
            #data: list of prompts
            #batch_size: integer indicating the size of a batch. WARNING: The batch size should be multiple of number of components!
            #operation_on_a_batch: It is a function that is applied on each batch.
            #output_handling: a function responsible for handling the output of each operation on a batch.
        if end_index <= start_index and end_index!=-1:
            print(f'Returning empty because it is full, thread_number={thread_number}')
            return []
        print(f'start_index:{start_index}, end_index={end_index}')
        list_of_outputs = []
        if batch_size%self.number_of_components!=0:
            print('ERROR!!!: The batch_size should dividable by the number of components we have.')
            return
        if end_index==-1:
            data = self.all_prompts_in_1D[start_index :]
        else:
            data = self.all_prompts_in_1D[start_index : end_index]
        data_size = len(data)
        number_of_batches = int(data_size/batch_size)+int(data_size%batch_size>0)
        
        for index in range( number_of_batches ):
            
            batch_index = index*batch_size
            if ( batch_index + batch_size > data_size ):
                batch = data[ batch_index : ]
            else:
                batch = data[ batch_index : batch_index + batch_size ]
            inputs_ids = self.tokenize_sentences( batch ).to(self.device_list[0]) #shape(batch_size , sentence_length)
            with open(log_file , 'a' ) as f:
                f.write(f'batch_number: {index}- thread num: {thread_number}\n')
                f.write(f'input_ids: {inputs_ids.size()}\n')

            print(f'batch_number: {index}- thread num: {thread_number}\n')
            starting_batch_input_len = inputs_ids.size(1)

            inputs_ids = torch.reshape(inputs_ids , (inputs_ids.size(0), inputs_ids.size(1) , 1) ) #shape(batch_size , input_len , 1)

            extra_added_paddings = torch.zeros( (inputs_ids.size(0) , 1 , inputs_ids.size(-1)) ) #This variable is added to address the changes to the number of padding tokens at the begginning of a sequence
            
            inputs_log_prob = torch.zeros( (inputs_ids.size(0) , 1 , inputs_ids.size(-1)) ).to(self.device_list[0]) #shape(barch_size , 1 , 1 )
            should_continue = True
            gen_respond_len = 0 #It is just for debugging
            start = time.time()
            print(f'Thread Num: {thread_number}')
            for i in self.device_list:
                print(f'Start of batch | max {i}:  , {torch.cuda.max_memory_allocated(i)} ')
            print('\n')
            number_of_clauses = 0
            current_clause_length = 0
            past_key_values = None
            max_clause_len = 50
            just_performed_ensemble = True # This variable shows if the ensemble is performed recently or not so that the model can continue and sample more tokens after the ensemble is performed
            while should_continue:
                gen_respond_len += 1
                start_iteration = time.time()
                # with open(log_file , 'a' ) as f:
                #     f.write(f'iteration: {gen_respond_len} - thread num: {thread_number}\n')

                inputs_ids_ready_infer = inputs_ids.transpose(1 , 2).reshape( (inputs_ids.size(0)*inputs_ids.size(-1), -1 ) ) # (batch_size*num_beam , input_len)
                inputs_ids_ready_infer, extra_added_paddings = self.shift_padding_to_start( inputs_ids_ready_infer , extra_added_paddings)
                inputs_ids = inputs_ids_ready_infer.reshape( (inputs_ids.size(0), inputs_ids.size(-1), -1 ) ).transpose(1 , 2)
            #     print( torchinfo.summary( self.model, input_data= inputs_ids_ready_infer ,batch_dim = 0, 
            # col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), device='cpu' , depth= 7) )

                batch_output , past_key_values, just_performed_ensemble, current_clause_length = self.get_next_word_score( inputs_ids_ready_infer , just_performed_ensemble, current_clause_length, past_key_values=past_key_values, top_k=top_k ) #(batch_size*num_beam , vocab_size)
                
                structured_batch_output = batch_output.reshape( ( inputs_ids.size(0) , -1 , self.vocab_size ) ).transpose(1 , 2) #(batch_size , vocab_size , num_beam)

                new_inputs_ids , new_inputs_log_prob, extra_added_paddings = self.output_handeling( structured_batch_output , inputs_ids, 
                                                                            inputs_log_prob, starting_batch_input_len , extra_added_paddings,
                                                                                 num_return_input = num_beam , top_k=top_k)

                gen_text_len = self.input_ids_to_gen_text_len(new_inputs_ids , starting_batch_input_len , extra_added_paddings) #shape(batch_size , 1 , num_candidate_beams)
                
                # print('Candidate beams:')
                # self.print_predictions( new_inputs_ids ,new_inputs_log_prob , starting_batch_input_len, extra_added_paddings)
                # print('\n')

                topk_object = torch.topk(-new_inputs_log_prob/(gen_text_len**0.1), k=num_beam , dim=-1 , largest=False )
                inputs_log_prob_divided = -topk_object.values # shape (batch_size , 1 , num_beam)

                topk_indices = topk_object.indices.reshape( (topk_object.indices.size(0),topk_object.indices.size(-1)) )
                batch_indices = self.make_batch_indice_tensor(batch_size = topk_indices.size(0) , expansion_size = topk_indices.size(-1))

                inputs_log_prob = new_inputs_log_prob[batch_indices,:,topk_indices].transpose( 1 , 2 ) #shape(batch_size , 1 , num_beam)
                
                inputs_ids = new_inputs_ids[batch_indices,:,topk_indices].transpose( 1 , 2 ) #shape(batch_size , input_len , num_beam)

                extra_added_paddings = extra_added_paddings[batch_indices, : , topk_indices.to('cpu')].transpose( 1 , 2 )#shape(batch_size , 1 , num_beam)
                # del new_inputs_ids; del new_inputs_log_prob; #torch.cuda.empty_cache()
                
                #To check if all the sequences have reached their padding token.
                pad_tensor = torch.tensor( [[ [self.tokenizer.pad_token_id] ]] )
                pad_expanded_tensor = pad_tensor.expand( ( inputs_ids.size(dim=0) , -1 , num_beam ) ).to(self.device_list[0])

                #To check if all the sequences have reached their EOS token.
                eos_tensor = torch.tensor( [[ [self.tokenizer.eos_token_id] ]] )
                eos_expanded_tensor = eos_tensor.expand( ( inputs_ids.size(dim=0) , -1 , num_beam ) ).to(self.device_list[0])

                #To check if now we need to perform ensemble:
                if all( torch.eq(inputs_ids[:,-1,:] , pad_expanded_tensor ).reshape((-1,)) ) or current_clause_length>=max_clause_len: #if we reached to the start of a clause or the end of a sequence for all the sequences.
                    # print('performing ensemble!')
                    inputs_ids , inputs_log_prob = self.ensemble_bleu( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings, batch )
                    just_performed_ensemble = True
                    past_key_values = None
                    number_of_clauses+=1

                if all( torch.eq(inputs_ids[:,-1,:] , eos_expanded_tensor ).reshape((-1,)) ):
                    should_continue = False
                # self.print_predictions( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings)
                # print('\n')
                
                if (inputs_ids.size(1)-starting_batch_input_len>=max_num_token):
                    # print('performing ensemble!')
                    should_continue = False
                    # inputs_ids , inputs_log_prob = self.ensemble_bleu( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings, batch )
                    number_of_clauses+=1

            most_probable_gen_text, gen_text_len_list = self.print_predictions( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings ,skip=True )
            with open(log_file , 'a' ) as f:
                f.write(f'\nThread_num:{thread_number} finished batch index: {index}\n')
                f.write(f'{time.time()-start} \t gen_text_len:{inputs_ids.size(1)-starting_batch_input_len}\n')
                f.write(f'\nThe chosen gen text:  {most_probable_gen_text}\n')
                f.write(f'number_of_gen_tokens:  {gen_text_len_list}\n')
                f.write(f'number_of_clauses:  {number_of_clauses}\n')
            del inputs_ids; del inputs_log_prob; del batch_output; #torch.cuda.empty_cache()
            print('The chosen gen text: ' , most_probable_gen_text, '\n')
            list_of_outputs.extend( most_probable_gen_text )
        print(f'Thread Num: {thread_number}')
        for i in self.device_list:
            print(f'End of process | max {i}:  , {torch.cuda.max_memory_allocated(i)} ')
        print('\n')
        return list_of_outputs
        
    def start_pipelines(self, batch_size , log_file , num_beam=1 , top_k=1 , max_num_token=300 ,  num_threads = 3 ):
        regular_thread_data_size = int(self.number_of_prompts/num_threads) * self.number_of_components
        thread_list = []
        init_output_list = self.get_predictions_form_logs( log_file , num_thread=num_threads) # this is for continuing any unfinished job. This will only work if the number of threads are equal to the number of threads used for the unifinished job
        for i in range(num_threads):
            num_processed_outputs = len(init_output_list[i])
            start_index = i * regular_thread_data_size
            if i == num_threads-1:
                my_thread = ThreadWithReturnValue(target= self.pipeline_process , args = (batch_size , log_file , num_beam , top_k , max_num_token ,start_index + num_processed_outputs * self.number_of_components , -1 , i) )
            else:
                my_thread = ThreadWithReturnValue(target= self.pipeline_process , args = (batch_size , log_file , num_beam , top_k , max_num_token , start_index + num_processed_outputs  * self.number_of_components , start_index + regular_thread_data_size , i ) )
            thread_list.append(my_thread)
            my_thread.start()

        list_of_outputs= []
        for i , my_thread in enumerate(thread_list):
            thread_output_list = my_thread.join()
            init_output_list[i].extend(thread_output_list)
            list_of_outputs.extend(init_output_list[i])
        
        return list_of_outputs
        
    def get_predictions_form_logs(self , file_name , num_thread=10):
        #getting the output_seqs from logs:
        output_list = [ [] for i in range(num_thread) ]
        if os.path.isfile(file_name)==False:
            return output_list
        
        with open(file_name , 'r') as f:
            lines = f.readlines()
        seen_thread = False
        for line in lines:
            if line.startswith('Thread_num:'):
                thread_index = int(line[11])
                seen_thread = True
            if seen_thread==True and line.startswith('The chosen gen text:  ['):
                batch_list = ast.literal_eval(line[22:-1])
                output_list[thread_index].extend(batch_list)
                seen_thread=False
                print('items in query:')
                print(batch_list)
                print('------------------------')
        for index , thread in enumerate(output_list):
            print(f'{index}- {len(thread)}')
        
        return output_list
        

import argparse       
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str)
    # parser.add_argument('--device', type = str, default='cpu')
    # parser.add_argument('--other_device', type = str, default='cpu')
    parser.add_argument('--output_file', type = str, default='./output_sequences.pkl')
    parser.add_argument('--log_file', type = str, default='./log.txt')
    
    parser.add_argument('--starting_index', type = int, default = 0)
    parser.add_argument('--ending_index', type = int, default = 100)

    parser.add_argument('--num_beam', type = int , default=5)
    parser.add_argument('--batch_size', type = int, default=1)
    parser.add_argument('--thread_num', type = int, default=4)

    opt = parser.parse_args()

    return opt

import requests
def send_notif(message):
    bot_token = "bot_token"
    chat_id = "chat_id"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    requests.post(url, data=payload)

if __name__ == "__main__":
    # accelerator = Accelerator()
    opt = parse_option()
    send_notif(f'{opt.output_file} is started in cedar:)')
    start_time = time.time()
    print(opt)
    # batch_size = opt.batch_size * 3
    # directory = './DAIL-SQL/dataset/process/'
    # dataset1 = 'SPIDER-TEST_SQL_0-SHOT_CTX-200_ANS-2048'
    # dataset2 = 'SPIDER-TEST_SQL_1-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048_llama_7b'
    # dataset3 = 'SPIDER-TEST_SQL_3-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048_llama_7b'
    # dataset_list = [dataset1 , dataset2 , dataset3]

    directory = './components/'
    dataset1 = 'SPIDER-TEST_SQL_3-SHOT_0-3_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset2 = 'SPIDER-TEST_SQL_3-SHOT_3-6_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset3 = 'SPIDER-TEST_SQL_3-SHOT_6-9_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset4 = 'SPIDER-TEST_SQL_3-SHOT_9-12_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset5 = 'SPIDER-TEST_SQL_3-SHOT_12-15_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset_list = [dataset1 , dataset2 , dataset3 , dataset4 , dataset5]
    batch_size = opt.batch_size * len(dataset_list)

    word_ensemble = LLM_Word_Level_Ensemble( opt.model_name )#, opt.device , opt.other_device)
    word_ensemble.load_prompts_from_datasets(directory , dataset_list , 
                                             starting_index= opt.starting_index, 
                                             ending_index= opt.ending_index)
    # list_of_outputs = word_ensemble.pipeline_process( batch_size , num_beam= opt.num_beam, 
    #                                                  top_k = opt.num_beam )
    list_of_outputs = word_ensemble.start_pipelines( batch_size , opt.log_file, num_beam=opt.num_beam , top_k=opt.num_beam , num_threads = opt.thread_num)
    with open(opt.output_file , 'wb') as f:
        pkl.dump(list_of_outputs , f)
    send_notif(f'**The task is done! output is being stored in cedar in{opt.output_file}\nProcess time:{time.time()-start_time}s')

