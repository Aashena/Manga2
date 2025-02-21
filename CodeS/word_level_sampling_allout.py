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
        self.device_list = [ 'cuda:0' ]
        print(f'loading the model into the GPUs: {self.device_list}')
        self.model = AutoModelForCausalLM.from_pretrained(model_name , torch_dtype = torch.float16 ).to(self.device_list[0])#{"": accelerator.process_index}) ###NO DEVICE
        print('Model successfully loaded')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side="left"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id # updating model config)

        
        
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
    
    def get_predictions(self, inputs , past_key_values_tensor=None):
        #performs the prediction on the tokenized sentences.
        #input:
            #torch tensor of shape(batch_size , length_of_sentences)
        #output: logits and next_cache
        with torch.no_grad():
            attention_mask = torch.logical_and(inputs != self.model.config.pad_token_id , inputs != self.model.config.eos_token_id).type(torch.int64)
            if past_key_values_tensor is not None:
                inputs = inputs[: , -1:]
                model_output = self.model( inputs , attention_mask=attention_mask, 
                use_cache=True, return_dict=True, past_key_values=torch.unbind(past_key_values_tensor,dim=0) )
            else:
                model_output = self.model( inputs , attention_mask=attention_mask, 
                use_cache=True, return_dict=True, past_key_values=None )

            return model_output['logits'][:,-1,:], torch.stack( model_output['past_key_values'] )

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
        
        word_probabilities = torch.nn.functional.softmax(
            word_scores, dim=1)
        word_probabilities = word_probabilities * ( word_scores != 0 )

        batch_indice = self.make_batch_indice_tensor( word_scores.size() , num_returned_words) #(batch_size , num_returned_words , num_beam)
        beam_indice = self.make_batch_indice_tensor( ( word_scores.size(-1), word_scores.size(0) ) , num_returned_words).transpose(0,2) #(batch_size , num_returned_words , num_beam)
        topk_candidates_prob = word_probabilities[ batch_indice , topk_candidates_indexes , beam_indice]#(batch_size , top-k , num_beam)
        
        # prob_vector , indice_vector = self.a( self.tokenizer.bos_token_id , topk_candidates_indexes , topk_candidates_prob)
        prob_vector , indice_vector = self.replace_token_with_eos( 5515 , topk_candidates_indexes , topk_candidates_prob)
        prob_vector , indice_vector = self.replace_token_with_eos( 2082 , topk_candidates_indexes , topk_candidates_prob)
        
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
    
    
    def get_next_word_score(self, sentences, past_key_values_tensor=None ,top_k=5):#operation_on_a_batch
        # Getting the word probabilities for the next word. It will have the probability for the top-k words, and it will zero out the rest. If the top_k argument is set to -1 then it will return the whole probability distribution.
        #Input:
            #sentences: torch tensor of size(batch size, input_len).
            #top_k: integer. Number of words with the highest probability that we want to have their probability back in the returning vecotr
        #Retrun:
            #torch tensor. The vector containing the probability of top_k words for the next word. The rest of the words are zeroed out. shape(batch_size , vocab_size)
        
        batch_size = len(sentences)
        next_token_candidates_tensor = torch.zeros( (batch_size , self.vocab_size) , dtype = torch.float16 ).to(self.device_list[-1])
        last_tokens = sentences[:,-1]
        active_texts_index = torch.nonzero( last_tokens!=self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')
        non_active_texts_index = torch.nonzero( last_tokens==self.tokenizer.eos_token_id ).reshape( (-1,) ).to('cpu')
        
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
        next_token_candidates_tensor[non_active_texts_index,:] = 0
        next_token_candidates_tensor[non_active_texts_index, self.tokenizer.eos_token_id ] = 1
        
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, k=top_k , dim=1 ).indices #shape( batch_size , top_k , num_beam)

        batch_indice_tensor = self.make_batch_indice_tensor(batch_size , top_k)

        masking_matrix = torch.zeros( ( next_token_candidates_tensor.size() ) ).to(self.device_list[-1])
        masking_matrix[ batch_indice_tensor , topk_candidates_indexes ] = 1
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_tensor_score = next_token_candidates_tensor * masking_matrix
        
        del sentences; torch.cuda.empty_cache()
        return topk_candidates_tensor_score , past_key_values_tensor #first output shape (batch_size , vocab_size)
        

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
                dataset_path = directory + dataset #+ '/' + 'questions.json'
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
        
    def ensemble(self, output_arrays ):
        #function for performing ensemble
        #input:
            #output_array: torch tensor representing the probability of each word in the matrix shape (number_of_ready_indexes_for_ensemble , number_of_components , vocab_size , num_beam)
        #return:
            #torch array containing the indice for top_k words in the ensembled vecotr probability: shape(number_of_ready_indexes_for_ensemble , top_k , num_beam)
            #torch array containing the probability of top_k words in the ensembled vector: shape (number_of_ready_indexes_for_ensemble , top_k , num_beam)

        # ensembled_output_arrays = output_arrays[:,-1,:,:] # shape ( number_of_ready_indexes_for_ensemble , vocab_size , num_beam) to get the output of 3-shot
        ensembled_output_arrays = output_arrays.mean(dim=1) # to get the average of word predictions
        # ensembled_output_arrays = output_arrays.sum(dim=1) # to get the ebbs result

        return ensembled_output_arrays

    
        
    # import torch.nn.functional as F
    def output_handeling(self, batch_output ,  input_ids , input_probs , starting_batch_input_len , num_return_input=1 , top_k=1):
        # This function handels the output of each batch. It performs post process and keeps the data in self.last_inferences so that it updates self.datasets_prompts_array. In the end of each step of inference on the data: 
        # input:
        #     batch_output: the output of operation_on_a_batch returns. a torch tensor shape (batch_size , vocab_size , num_beams ) it is on self.other_device
            # input_ids: The input ids of the model that predicted the batch_output (batch_size , input_length , num_beams) 
            # input_probs: contains the probability of each input sequence (batch_size , 1 , num_beam)
        # return:
        #     new input_ids that now the newly predicted tokens are added to them on self.device_list[0]
             #A flag showing if the sampling is done or not.
        batch_size = batch_output.size(dim=0)
        num_beam = batch_output.size(dim=-1)
        number_of_processing_prompts = int( batch_size / self.number_of_components )
        
        ready_to_ensemble = torch.reshape( batch_output , (number_of_processing_prompts , self.number_of_components , self.vocab_size , -1) ) #shape(number_of_processing_prompts , number_of_components , vocab_size , num_beam)
        
        ensembled_scores = self.ensemble( ready_to_ensemble ) #shape (number_of_processing_prompts , vocab_size , num_beam)

        
        new_words_vector, new_words_prob  = self.get_top_words_id_prob_from_score_vector( ensembled_scores , num_returned_words=num_return_input) #shape (number_of_processing_prompts , num_returned_words , num_beam)
        # print(new_words_vector , '\n')
        new_words_vector= new_words_vector.repeat_interleave(repeats= self.number_of_components ,dim=0).reshape(-1, 1, num_return_input*num_beam).to(self.device_list[0])  # shape(batch_size , 1 , num_beam*num_return_input) batch_size = number_of_processing_prompts*number_of_components

        input_ids = input_ids.repeat( 1,1,num_return_input) #shape (batch_size , input_len , num_beam*num_return_input)
        
        token_mask = (input_ids[:,-1:,:]!=self.tokenizer.eos_token_id)
        token_unmask = (input_ids[:,-1:,:]==self.tokenizer.eos_token_id) * self.tokenizer.eos_token_id
        new_words_vector = new_words_vector * token_mask + token_unmask
        
        new_input_ids = torch.cat( ( input_ids , new_words_vector ) , dim=1 ) #shape (batch_size , input_length+1 , num_beam*num_return_input)
        
        new_words_prob = new_words_prob.repeat_interleave(repeats= self.number_of_components ,dim=0).reshape(-1,1,num_return_input*num_beam).to(self.device_list[0])  # shape(batch_size , 1 , num_beam*num_return_input)

        input_probs = input_probs.repeat(1,1,num_return_input) #shape (batch_size , 1 , num_beam*num_return_input)

        eos_masking = self.input_ids_to_gen_text_len( new_input_ids , starting_batch_input_len) == (new_input_ids.size(1)-starting_batch_input_len)
        bad_candidates_masking = (new_words_prob == 0)
        # print(bad_candidates_masking)
        new_input_probs = input_probs + torch.log(new_words_prob) * eos_masking + -100 * bad_candidates_masking #shape (batch_size , 1 , num_beam*num_return_input)

        del input_ids; del input_probs; del batch_output; #torch.cuda.empty_cache()
        
        return new_input_ids , new_input_probs #(batch_size , input_len , num_candidate_beams ) and (batch_size , 1 , num_candidate_beams ) num_candidate_beams = num_beam*num_return_input
            
    def print_predictions(self, inputs_ids , input_prob=None , starting_batch_input_len=0 ,  skip = False):
        #input_ids: (batch_size , input_len , num_beam)
        if input_prob is None:
            input_prob = torch.zeros( (inputs_ids.size(0) , 1 , inputs_ids.size(-1) ) )
        gen_text_list = []
        for i  in range( 0 , inputs_ids.size(0) , self.number_of_components ):
            # print(f'Index in batch: {int( (i+1)/self.number_of_components )}')
            input_text_list = self.tokenizer.batch_decode( inputs_ids[i,starting_batch_input_len:,:].transpose( 0,1 ) ,
                                                          skip_special_tokens=skip )
            gen_text_list.append( input_text_list[0] )
            for j in range( len(input_text_list) ):
                print(f'Beam number {j}, prob={input_prob[i,0,j]} --> {input_text_list[j]} --> {inputs_ids[i,-1,j]}')
        return gen_text_list


    def input_ids_to_gen_text_len(self , input_ids , starting_batch_input_len):
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
        return end_idx_batch_input.add(-starting_batch_input_len).to(self.device_list[0])
        
        
        
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
        list_of_input_ids = []
        list_of_inputs_log_prob = []
        list_of_starting_batch_input_len = []
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
            inputs_log_prob = torch.zeros( (inputs_ids.size(0) , 1 , inputs_ids.size(-1)) ).to(self.device_list[0]) #shape(barch_size , 1 , 1 )
            should_continue = True
            gen_respond_len = 0 #It is just for debugging
            start = time.time()
            print(f'Thread Num: {thread_number}')
            for i in self.device_list:
                print(f'Start of batch | max {i}:  , {torch.cuda.max_memory_allocated(i)} ')
            print('\n')
            print(inputs_ids.size())
            input_length = inputs_ids.size(1)

            #block of code to replicate the original sampling
            # token_ids_of_example_sql = self.tokenizer("SELECT * FROM tables ;")["input_ids"]
            # print(token_ids_of_example_sql)
            # if token_ids_of_example_sql[-1] == self.tokenizer.eos_token_id:
            #     new_eos_token_id = token_ids_of_example_sql[-2]
            # else:
            #     new_eos_token_id = token_ids_of_example_sql[-1]
            # self.model.config.eos_token_id = new_eos_token_id
            # self.tokenizer.eos_token_id = new_eos_token_id
            # print("new_eos_token_id:", new_eos_token_id)
            # print("tokenizer.decode(new_eos_token_id): '{}'".format(self.tokenizer.decode(new_eos_token_id)))
            # input_ids = self.tokenizer(batch[0] , truncation = False)["input_ids"]
            # attention_mask = [1] * len(input_ids)
            # inputs = {
            #     "input_ids": torch.tensor([input_ids]).to(self.device_list[0]), # torch.int64
            #     "attention_mask": torch.tensor([attention_mask]).to(self.device_list[0]) # torch.int64
            # }
            # with torch.no_grad():
            #     generate_ids = self.model.generate(
            #         **inputs,
            #         max_new_tokens = 256,
            #         num_beams = 4,
            #         num_return_sequences = 4,
            #         use_cache = True,
            #         eos_token_id = new_eos_token_id,
            #     )

            # generated_sqls = self.tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
            # for sql in generated_sqls:
            #     print('generated_sqls: ' , sql)
            #     print('------------------------')
            past_key_values_tensor = None
            is_first_token = True
            while should_continue:
                gen_respond_len += 1
                start_iteration = time.time()
                # with open(log_file , 'a' ) as f:
                #     f.write(f'iteration: {gen_respond_len} - thread num: {thread_number}\n')

                inputs_ids_ready_infer = inputs_ids.transpose(1 , 2).reshape( (inputs_ids.size(0)*inputs_ids.size(-1), -1 ) ) # (batch_size*num_beam , input_len)

            #     print( torchinfo.summary( self.model, input_data= inputs_ids_ready_infer ,batch_dim = 0, 
            # col_names = ('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), device='cpu' , depth= 7) )
                
                batch_output , past_key_values_tensor = self.get_next_word_score( inputs_ids_ready_infer , past_key_values_tensor=past_key_values_tensor, top_k=top_k ) #(batch_size*num_beam , vocab_size)

                structured_batch_output = batch_output.reshape( ( inputs_ids.size(0) , -1 , self.vocab_size ) ).transpose(1 , 2) #(batch_size , vocab_size , num_beam)

                new_inputs_ids , new_inputs_log_prob = self.output_handeling( structured_batch_output , inputs_ids, 
                                                                            inputs_log_prob, starting_batch_input_len ,
                                                                                 num_return_input = num_beam , top_k=top_k)

                gen_text_len = self.input_ids_to_gen_text_len(new_inputs_ids , starting_batch_input_len) #shape(batch_size , 1 , num_candidate_beams)
                
                # print('Candidate beams:')
                # self.print_predictions( new_inputs_ids ,new_inputs_log_prob/gen_text_len , starting_batch_input_len)
                # print('\n')

                topk_object = torch.topk(-new_inputs_log_prob/(gen_text_len**0.5), k=num_beam , dim=-1 , largest=False )
                inputs_log_prob_divided = -topk_object.values # shape (batch_size , 1 , num_beam)
                topk_indices = topk_object.indices.reshape( (topk_object.indices.size(0),topk_object.indices.size(-1)) ).to('cpu')
                
                batch_indices = self.make_batch_indice_tensor(batch_size = topk_indices.size(0) , expansion_size = topk_indices.size(-1))
                # print('topk_indices: ' , topk_indices)
                # print('batch_indices: ' , batch_indices)
                inputs_log_prob = new_inputs_log_prob[batch_indices,:,topk_indices].transpose( 1 , 2 ) #shape(batch_size , 1 , num_beam)
                # print('past_key_values[0].size(): ' , past_key_values[0].size())
                inputs_ids = new_inputs_ids[batch_indices,:,topk_indices].transpose( 1 , 2 ) #shape(batch_size , input_len , num_beam)
                if is_first_token:
                    selected_beams_indice = ( ( batch_indices ) + (topk_indices/num_beam) ).to(torch.int32).reshape((-1,))
                else:
                    selected_beams_indice = ( (batch_indices * num_beam) + torch.remainder(topk_indices, num_beam) ).to(torch.int32).reshape((-1,))
                # print('selected_beams_indice: ' , selected_beams_indice)
                
                past_key_values_tensor = past_key_values_tensor[: , selected_beams_indice , : , :]

                del new_inputs_ids; del new_inputs_log_prob; #torch.cuda.empty_cache()
                
                # self.print_predictions( inputs_ids , inputs_log_prob , starting_batch_input_len)
                # print('\n')

                #To check if all the sequences have reached their EOS token.
                eos_tensor = torch.tensor( [[ [self.tokenizer.eos_token_id] ]] )
                eos_expanded_tensor = eos_tensor.expand( ( inputs_ids.size(dim=0) , -1 , num_beam ) ).to(self.device_list[0])
                
                if all( torch.eq(inputs_ids[:,-1,:] , eos_expanded_tensor ).reshape((-1,)) ):
                    should_continue = False
                elif inputs_ids.size(1)-starting_batch_input_len>=max_num_token:
                    should_continue = False
                else:
                    should_continue = True
                is_first_token = False
            most_probable_gen_text = self.print_predictions( inputs_ids , inputs_log_prob , starting_batch_input_len ,skip=True )
            list_of_input_ids.extend(inputs_ids)
            list_of_inputs_log_prob.extend(inputs_log_prob)
            list_of_starting_batch_input_len.append(starting_batch_input_len)
            list_of_batch_text.extend(batch)
            with open(log_file , 'a' ) as f:
                f.write(f'\nThread_num:{thread_number} finished batch index: {index}\n')
                f.write(f'{time.time()-start} \t gen_text_len:{inputs_ids.size(1)-starting_batch_input_len}\n')
                f.write(f'\nThe chosen gen text:  {most_probable_gen_text}\n')
            del inputs_ids; del inputs_log_prob; del batch_output; #torch.cuda.empty_cache()
            print('The chosen gen text: ' , most_probable_gen_text, '\n')
            list_of_outputs.extend( most_probable_gen_text )
        print(f'Thread Num: {thread_number}')
        for i in self.device_list:
            print(f'End of process | max {i}:  , {torch.cuda.max_memory_allocated(i)} ')
        print('\n')
        return list_of_outputs, list_of_input_ids, list_of_inputs_log_prob, list_of_starting_batch_input_len, list_of_batch_text
        
    def start_pipelines(self, batch_size , log_file , num_beam=1 , top_k=1 , max_num_token=200 ,  num_threads = 3):
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
        list_of_input_ids = []
        list_of_inputs_log_prob = []
        list_of_starting_batch_input_len = []
        list_of_batch_text = []
        for i , my_thread in enumerate(thread_list):
            thread_output_list, thread_input_ids_list, thread_inputs_log_prob_list, thread_starting_batch_input_len_list, thread_batch_text_list = my_thread.join()
            init_output_list[i].extend(thread_output_list)
            
            list_of_input_ids.extend(thread_input_ids_list)
            list_of_inputs_log_prob.extend(thread_inputs_log_prob_list)
            list_of_starting_batch_input_len.extend(thread_starting_batch_input_len_list)
            list_of_batch_text.extend(thread_batch_text_list)

            list_of_outputs.extend(init_output_list[i])
        
        return list_of_outputs, list_of_input_ids, list_of_inputs_log_prob, list_of_starting_batch_input_len, list_of_batch_text
        
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
    bot_token = "7490837017:AAGIuWfuP7OpdKN2o-cy6lyhnnsdL-1PRZo"
    chat_id = "294248322"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    requests.post(url, data=payload)

if __name__ == "__main__":
    # accelerator = Accelerator()
    opt = parse_option()
    send_notif('The job is started in manga-3:)')
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
    
    # dataset2 = 'SPIDER-TEST_SQL_3-SHOT_3-6_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset3 = 'SPIDER-TEST_SQL_3-SHOT_6-9_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset4 = 'SPIDER-TEST_SQL_3-SHOT_9-12_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset5 = 'SPIDER-TEST_SQL_3-SHOT_12-15_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset_list = [dataset1 , dataset2 , dataset3 , dataset4 , dataset5]
    batch_size = opt.batch_size * len(dataset_list)

    word_ensemble = LLM_Word_Level_Ensemble(opt.model_name)#, opt.device , opt.other_device)
    word_ensemble.load_prompts_from_datasets(directory , dataset_list , 
                                             starting_index= opt.starting_index, 
                                             ending_index= opt.ending_index)
    # list_of_outputs = word_ensemble.pipeline_process( batch_size , num_beam= opt.num_beam, 
    #                                                  top_k = opt.num_beam )
    list_of_outputs, list_of_input_ids, list_of_inputs_log_prob, list_of_starting_batch_input_len, list_of_batch_text = word_ensemble.start_pipelines( batch_size , opt.log_file, num_beam=opt.num_beam , top_k=opt.num_beam , num_threads = opt.thread_num)
    with open(opt.output_file , 'wb') as f:
        pkl.dump(list_of_outputs , f)
    with open(opt.output_file[:-4] + '_input_ids.pkl' , 'wb') as f:
        pkl.dump(list_of_input_ids , f)
    with open(opt.output_file[:-4] + '_inputs_log_prob.pkl' , 'wb') as f:
        pkl.dump(list_of_inputs_log_prob , f)
    with open(opt.output_file[:-4] + '_starting_batch_input_len.pkl' , 'wb') as f:
        pkl.dump(list_of_starting_batch_input_len , f)
    with open(opt.output_file[:-4] + '_batch_text.pkl' , 'wb') as f:
        pkl.dump(list_of_batch_text , f)
    send_notif(f'The task is done! output is being stored in manga-3 in{opt.output_file}\nProcess time:{time.time()-start_time}s')

