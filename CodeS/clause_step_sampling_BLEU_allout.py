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
        seed=25
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.device_list = ['cuda:0']
        print(f'loading the model into two GPUs: {self.device_list}')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16).to(self.device_list[0])#{"": accelerator.process_index}) ###NO DEVICE
        print('Model successfully loaded')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side="left"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id # updating model config)
        self.just_performed_ensemble = True # This variable shows if the ensemble is performed recently or not so that the model can continue and sample more tokens after the ensemble is performed
        
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

    
    def get_predictions(self, inputs , past_key_values_tensor=None , num_beam=4):
        #performs the prediction on the tokenized sentences.
        #input:
            #torch tensor of shape(batch_size , length_of_sentences)
        #output: logits and next_cache
        # print(f'before any predictions | max:  , {torch.cuda.max_memory_allocated(self.device_list[0])} ')
        # print(f'before any predictions | real:  , {torch.cuda.memory_allocated(self.device_list[0])} ')
        with torch.no_grad():
            attention_mask = (inputs != self.model.config.pad_token_id).type(torch.int64)
            if past_key_values_tensor is not None:
                inputs = inputs[: , -1:]
                model_output = self.model( inputs , attention_mask=attention_mask, 
                use_cache=True, return_dict=True, past_key_values=torch.unbind(past_key_values_tensor,dim=0) )
                # print("model_output['logits'][:,-1,:].size() in get_predictions(): " , model_output['logits'][:,-1,:].size())
                return model_output['logits'][:,-1,:], torch.stack( model_output['past_key_values'] )
            else:
                # print( 'inputs.size() in get_predictions():' , inputs.size())
                for i in range(0, inputs.size(0) , num_beam):
                    if i + num_beam>= inputs.size(0):
                        attention_mask = (inputs[ i: , : ] != self.model.config.pad_token_id).type(torch.int64)
                        position_ids = torch.cumsum(attention_mask, dim=-1)
                        model_output = self.model( inputs[ i: , : ] , attention_mask=attention_mask, position_ids=position_ids,
                            use_cache=True, return_dict=True, past_key_values=None )
                    else:
                        attention_mask = (inputs[ i:i+num_beam , : ] != self.model.config.pad_token_id).type(torch.int64)
                        position_ids = torch.cumsum(attention_mask, dim=-1)
                        model_output = self.model( inputs[ i:i+num_beam , : ] , attention_mask=attention_mask, position_ids=position_ids,
                            use_cache=True, return_dict=True, past_key_values=None )
                    if past_key_values_tensor == None:
                        model_logits = model_output['logits'][:,-1,:]
                        past_key_values_tensor = torch.stack( model_output['past_key_values'] )
                    else:
                        model_logits = torch.cat( [model_logits  ,model_output['logits'][:,-1,:]] ,dim=0 )
                        past_key_values_tensor = torch.cat( [past_key_values_tensor, torch.stack( model_output['past_key_values'] ) ],
                                                            dim=1 )
                    
                    # print('model_logits.size() in get_predictions(): ' , model_logits.size())
                return model_logits, past_key_values_tensor
            # print(f'After any predictions | max:  , {torch.cuda.max_memory_allocated(self.device_list[0])} ')
            # print(f'After any predictions | real:  , {torch.cuda.memory_allocated(self.device_list[0])} ')
            

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
        prob_vector , indice_vector = self.replace_token_with_eos( 2082 , indice_vector , prob_vector)

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
        sql_keywords = [
            "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET",
             "UNION", "WITH" , "INTERSECT" , "EXCEPT"
        ]
        decoded_sentences = self.tokenizer.batch_decode( sentences , skip_special_tokens=False )
        output_list = []
        for sent in decoded_sentences:
            found_keyword = False
            for keyword in sql_keywords:
                if sent.upper()[-(len(keyword)):] == keyword:
                    found_keyword = True
                    # print('found keyword')
                    break
            output_list.append(found_keyword)
        return torch.tensor( output_list ).to(sentences.device)

    def get_next_word_score(self, sentences, just_performed_ensemble, current_clause_length,
                             current_clause_num,  past_key_values_tensor=None ,top_k=5):#operation_on_a_batch
        # Getting the word probabilities for the next word. It will have the probability for the top-k words, and it will zero out the rest. If the top_k argument is set to -1 then it will return the whole probability distribution.
        #Input:
            #sentences: torch tensor of size(batch size, input_len).
            #top_k: integer. Number of words with the highest probability that we want to have their probability back in the returning vecotr
        #Retrun:
            #torch tensor. The vector containing the probability of top_k words for the next word. The rest of the words are zeroed out. shape(batch_size , vocab_size)
        
        batch_size = len(sentences)
        # print('batch_size: ' , batch_size)
        next_token_candidates_tensor = torch.zeros( (batch_size , self.vocab_size) , dtype = torch.float16 ).to(self.device_list[-1])
        last_tokens = sentences[:,-1]
        
        non_active_texts_index = torch.nonzero( last_tokens==self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')

        next_token_candidates_tensor[non_active_texts_index,:] = 0
        next_token_candidates_tensor[non_active_texts_index, self.tokenizer.eos_token_id ] = 1
        
        if just_performed_ensemble == False:
            current_clause_length +=1
            is_waiting = self.check_if_start_of_clause(sentences)
            is_active = torch.logical_and( torch.logical_not(is_waiting) , last_tokens!=self.tokenizer.eos_token_id )
            if False not in is_active:
                current_clause_length = 5 #any low number above 4
            active_texts_index = torch.nonzero( is_active ).reshape( (-1,) ).to('cpu')
            reached_end_of_clause = torch.nonzero( is_waiting ).reshape( (-1,) ).to('cpu')
            next_token_candidates_tensor[reached_end_of_clause,:] = 0
            next_token_candidates_tensor[reached_end_of_clause, self.tokenizer.pad_token_id ] = 1
            # waiting_texts_index = torch.nonzero( last_tokens==self.tokenizer.pad_token_id ).reshape( (-1,) ).to('cpu')
            # print( 'waiting_texts_index: ' , waiting_texts_index )
            # next_token_candidates_tensor[waiting_texts_index,:] = 0
            # next_token_candidates_tensor[waiting_texts_index, self.tokenizer.pad_token_id ] = 1
        # elif True in self.check_if_start_of_clause(sentences):
        else:
            if current_clause_num == 0:
                current_clause_length +=1
                active_texts_index = torch.nonzero( last_tokens!=self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')
                if current_clause_length>4:
                    just_performed_ensemble = False
            else:
                current_clause_length = 1
                active_texts_index = torch.nonzero( last_tokens!=self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')
                just_performed_ensemble = False
        # else:
        #     active_texts_index = torch.nonzero( last_tokens!=self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')

        if past_key_values_tensor is not None:
            tmp_past_key_values_tensor = past_key_values_tensor[ :, active_texts_index , : ,:]
        else:
            tmp_past_key_values_tensor=None


        if active_texts_index.size(0)>0:
            logits , tmp_past_key_values_tensor = self.get_predictions(sentences[active_texts_index,:] , 
                                                            past_key_values_tensor= tmp_past_key_values_tensor) #shape(batch_size , vocab_size)
            
            past_key_values_tensor_size = (tmp_past_key_values_tensor.size(0), sentences.size(0), 
                                           tmp_past_key_values_tensor.size(2), tmp_past_key_values_tensor.size(3) )
            past_key_values_tensor = torch.zeros( past_key_values_tensor_size ,dtype=torch.float16 ).to(self.device_list[0])
            past_key_values_tensor[: , active_texts_index , : , :] = tmp_past_key_values_tensor

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
        return topk_candidates_tensor_score , past_key_values_tensor, just_performed_ensemble, current_clause_length #first output shape (batch_size , vocab_size)
        

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
        
    def ensemble_bleu(self, inputs_ids , inputs_log_prob , starting_batch_input_len , extra_added_paddings ):
        #function for performing ensemble using the bleu metric between the candidate sequences.
        #input:
            #inputs_ids: torch tensor representing the prompt tokens per component with shape (batch_size , input_len , num_beam)
            #inputs_log_prob: torch tensor representing the probability of each input_ids with the shape(batch_size , 1 , num_beam)
            #starting_batch_input_len: The length of the batch before any predictions.
        #return:
            #ensembled_inputs_ids: torch tensor with size (batch_size , input_len , num_beam)
            #ensembled_inputs_log_prob: torch tensor with size (batch_size , 1 , num_beam)
        # print('extra_added_paddings before ensemble: ' , extra_added_paddings)
        # print(f'start of ensemble | max:  , {torch.cuda.max_memory_allocated(self.device_list[0])} ')
        batch_size = inputs_ids.size( dim=0 )
        num_beam = inputs_ids.size( dim=-1 )
        ensembled_inputs_ids = inputs_ids.clone()
        ensembled_inputs_log_prob = inputs_log_prob.clone()
        for i in range ( 0, batch_size ,self.number_of_components ): #number_of_processing_prompts
            tokenized_responses = []
            decoded_text_list = []
            last_component = i + self.number_of_components
            #Tokenizing the candidate sequences
            for j in range(num_beam):
                decoded_text_list.extend( self.tokenizer.batch_decode( inputs_ids[ i:last_component , starting_batch_input_len: , j ] ,
                                            skip_special_tokens=True ) )
            for text in decoded_text_list: #number of candidates we have for each question
                tokenized_responses.append( word_tokenize( text ) )
            blue_scores = []
            #Calculating the bleu metrics for each candidate
            for j in range( len( tokenized_responses ) ): #[tok_component1_beam1, tok_component2_beam1, tok_component3_beam1, ..., tok_component1_beam2, tok_component2_beam2 , ...]
                temp_tokenized_responses = tokenized_responses.copy()
                tokenized_response = temp_tokenized_responses.pop(j)
                blue_score = sentence_bleu( temp_tokenized_responses , tokenized_response )
                blue_scores.append(blue_score)
            selected_candidate_list = [] #containing tuples like (component_index , beam_index)
            selected_candidate_ex_add_tok = [] #It shows the number of extra added tokens for each of the selected candidates
            #Finding the sequence with the highest bleu score.
            for beam in range(num_beam):
                max_bleu_score_value = max( blue_scores )
                max_index_bleu_score = blue_scores.index(max_bleu_score_value)
                blue_scores[ max_index_bleu_score ] = -100
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
                    ensembled_inputs_log_prob[i+component_index , 0 , beam_index] = inputs_log_prob[selected_candidate_list[beam_index][0]+i,
                                                                                    0, selected_candidate_list[beam_index][1]]
        # print('extra_added_paddings after ensemble: ' , extra_added_paddings)
        # print(f'end of ensemble | max:  , {torch.cuda.max_memory_allocated(self.device_list[0])} ')
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
        list_of_inputs_log_prob = []
        list_of_batch_text = []
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
            past_key_values_tensor = None
            max_clause_len = 50
            just_performed_ensemble = True # This variable shows if the ensemble is performed recently or not so that the model can continue and sample more tokens after the ensemble is performed
            while should_continue:
                gen_respond_len += 1
                start_iteration = time.time()
                # with open(log_file , 'a' ) as f:
                #     f.write(f'iteration: {gen_respond_len} - thread num: {thread_number}\n')

                inputs_ids_ready_infer = inputs_ids.transpose(1 , 2).reshape( (inputs_ids.size(0)*inputs_ids.size(-1), -1 ) ) # (batch_size*num_beam , input_len) [beam1_comp1, beam2_comp1, ..., beam1_comp2, ..., beamN_compN]
                inputs_ids_ready_infer, extra_added_paddings = self.shift_padding_to_start( inputs_ids_ready_infer , extra_added_paddings)
                inputs_ids = inputs_ids_ready_infer.reshape( (inputs_ids.size(0), inputs_ids.size(-1), -1 ) ).transpose(1 , 2)
            #     print( torchinfo.summary( self.model, input_data= inputs_ids_ready_infer ,batch_dim = 0, 
            # col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), device='cpu' , depth= 7) )
                
                batch_output , past_key_values_tensor, just_performed_ensemble, current_clause_length = self.get_next_word_score( inputs_ids_ready_infer , just_performed_ensemble, 
                                                                 current_clause_length, current_clause_num=number_of_clauses ,
                                                                 past_key_values_tensor=past_key_values_tensor, top_k=top_k ) #(batch_size*num_beam , vocab_size)

                structured_batch_output = batch_output.reshape( ( inputs_ids.size(0) , -1 , self.vocab_size ) ).transpose(1 , 2) #(batch_size , vocab_size , num_beam)

                new_inputs_ids , new_inputs_log_prob, extra_added_paddings = self.output_handeling( structured_batch_output , inputs_ids, 
                                                                            inputs_log_prob, starting_batch_input_len , extra_added_paddings,
                                                                                 num_return_input = num_beam , top_k=top_k) #[candid1_beam1_comp1, candid1_beam2_comp1, candid1_beam3_comp1, ..., candidN_beam1_comp1, candidN_beam2_comp1, ... , candidN_beamN_compN]

                gen_text_len = self.input_ids_to_gen_text_len(new_inputs_ids , starting_batch_input_len , extra_added_paddings) #shape(batch_size , 1 , num_candidate_beams)
                
                # print('Candidate beams:')
                # self.print_predictions( new_inputs_ids ,new_inputs_log_prob/(gen_text_len**0.5) , starting_batch_input_len, extra_added_paddings)
                # print('\n')

                topk_object = torch.topk(-new_inputs_log_prob/(gen_text_len**0.5), k=num_beam , dim=-1 , largest=False )
                inputs_log_prob_divided = -topk_object.values # shape (batch_size , 1 , num_beam)

                topk_indices = topk_object.indices.reshape( (topk_object.indices.size(0),topk_object.indices.size(-1)) ).to('cpu')
                batch_indices = self.make_batch_indice_tensor(batch_size = topk_indices.size(0) , expansion_size = topk_indices.size(-1))

                inputs_log_prob = new_inputs_log_prob[batch_indices,:,topk_indices].transpose( 1 , 2 ) #shape(batch_size , 1 , num_beam)
                
                inputs_ids = new_inputs_ids[batch_indices,:,topk_indices].transpose( 1 , 2 ) #shape(batch_size , input_len , num_beam)

                extra_added_paddings = extra_added_paddings[batch_indices, : , topk_indices.to('cpu')].transpose( 1 , 2 )#shape(batch_size , 1 , num_beam)
                # print('topk_indices: ' , topk_indices)
                # print('batch_indices: ' , batch_indices)
                # print('number_of_clauses: ' , number_of_clauses)
                # print('current_clause_length: ' , current_clause_length)
                if number_of_clauses==0 and current_clause_length==1:
                    selected_beams_indice = ( ( batch_indices ) + (topk_indices/num_beam) ).to(torch.int32).reshape((-1,))
                else:
                    selected_beams_indice = ( (batch_indices * num_beam) + topk_indices%num_beam ).to(torch.int32).reshape((-1,))
                # print('selected_beams_indice: ' , selected_beams_indice)
                
                past_key_values_tensor = past_key_values_tensor[: , selected_beams_indice , : , :]

                # del new_inputs_ids; del new_inputs_log_prob; #torch.cuda.empty_cache()
                
                #To check if all the sequences have reached their padding token.
                pad_tensor = torch.tensor( [[ [self.tokenizer.pad_token_id] ]] )
                pad_expanded_tensor = pad_tensor.expand( ( inputs_ids.size(dim=0) , -1 , num_beam ) ).to(self.device_list[0])

                #To check if all the sequences have reached their EOS token.
                eos_tensor = torch.tensor( [[ [self.tokenizer.eos_token_id] ]] )
                eos_expanded_tensor = eos_tensor.expand( ( inputs_ids.size(dim=0) , -1 , num_beam ) ).to(self.device_list[0])

                #To check if now we need to perform ensemble:
                #To check if now we need to perform ensemble:
                if all( torch.eq(inputs_ids[:,-1,:] , pad_expanded_tensor ).reshape((-1,)) ) or current_clause_length>=max_clause_len: #if we reached to the start of a clause or the end of a sequence for all the sequences.
                    # print('performing ensemble!')
                    inputs_ids , inputs_log_prob = self.ensemble_bleu( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings )
                    just_performed_ensemble = True
                    past_key_values_tensor = None
                    number_of_clauses+=1

                if all( torch.eq(inputs_ids[:,-1,:] , eos_expanded_tensor ).reshape((-1,)) ):
                    should_continue = False
                    # inputs_ids , inputs_log_prob = self.ensemble_bleu( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings )
                    past_key_values_tensor = None
                    number_of_clauses+=1

                # self.print_predictions( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings)
                # print('\n')
                
                if (inputs_ids.size(1)-starting_batch_input_len>=max_num_token):
                    # print('performing ensemble!')
                    should_continue = False
                    # inputs_ids , inputs_log_prob = self.ensemble_bleu( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings)
                    past_key_values_tensor = None
                    number_of_clauses+=1

            most_probable_gen_text, gen_text_len_list = self.print_predictions( inputs_ids , inputs_log_prob , starting_batch_input_len, extra_added_paddings ,skip=True )
            list_of_inputs_log_prob.extend(inputs_log_prob)
            list_of_batch_text.extend(batch)
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
        return list_of_outputs, list_of_inputs_log_prob, list_of_batch_text
        
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
        list_of_inputs_log_prob = []
        list_of_batch_text = []
        for i , my_thread in enumerate(thread_list):
            thread_output_list, thread_inputs_log_prob_list, thread_batch_text_list = my_thread.join()
            init_output_list[i].extend(thread_output_list)

            list_of_inputs_log_prob.extend(thread_inputs_log_prob_list)
            list_of_batch_text.extend(thread_batch_text_list)

            list_of_outputs.extend(init_output_list[i])
        
        return list_of_outputs, list_of_inputs_log_prob, list_of_batch_text
        
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
    send_notif('The job is started in manga-2:)')
    start_time = time.time()
    print(opt)
    # batch_size = opt.batch_size * 3
    # directory = './DAIL-SQL/dataset/process/'
    # dataset1 = 'SPIDER-TEST_SQL_0-SHOT_CTX-200_ANS-2048'
    # dataset2 = 'SPIDER-TEST_SQL_1-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048_llama_7b'
    # dataset3 = 'SPIDER-TEST_SQL_3-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048_llama_7b'
    # dataset_list = [dataset1 , dataset2 , dataset3]

    directory = './components/'
    dataset1 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_0-5_max_tokens_8192_max_new_tokens_256.json'
    dataset2 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_5-10_max_tokens_8192_max_new_tokens_256.json'
    dataset3 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_10-15_max_tokens_8192_max_new_tokens_256.json'
    dataset4 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_15-20_max_tokens_8192_max_new_tokens_256.json'
    dataset5 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_20-25_max_tokens_8192_max_new_tokens_256.json'
    dataset_list = [dataset1 , dataset2 , dataset3 , dataset4 , dataset5]
    batch_size = opt.batch_size * len(dataset_list)

    word_ensemble = LLM_Word_Level_Ensemble( opt.model_name )#, opt.device , opt.other_device)
    word_ensemble.load_prompts_from_datasets(directory , dataset_list , 
                                             starting_index= opt.starting_index, 
                                             ending_index= opt.ending_index)
    # list_of_outputs = word_ensemble.pipeline_process( batch_size , num_beam= opt.num_beam, 
    #                                                  top_k = opt.num_beam )
    list_of_outputs, list_of_inputs_log_prob, list_of_batch_text = word_ensemble.start_pipelines( batch_size , opt.log_file, num_beam=opt.num_beam , top_k=opt.num_beam , num_threads = opt.thread_num)
    with open(opt.output_file , 'wb') as f:
        pkl.dump(list_of_outputs , f)
    with open(opt.output_file[:-4] + '_inputs_log_prob.pkl' , 'wb') as f:
        pkl.dump(list_of_inputs_log_prob , f)
    with open(opt.output_file[:-4] + '_batch_text.pkl' , 'wb') as f:
        pkl.dump(list_of_batch_text , f)
    send_notif(f'The task is done! output is being stored in manga-2 in{opt.output_file}\nProcess time:{time.time()-start_time}s')

