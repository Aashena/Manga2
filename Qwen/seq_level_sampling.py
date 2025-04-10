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

#These lines are added for running "onekq-ai/OneSQL-v0.1-Qwen-32B"
# from peft import PeftModel

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
        self.device_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
            
        print(f'loading the model into the GPUs: {self.device_list}')

        #These lines are added for running "onekq-ai/OneSQL-v0.1-Qwen-32B"
        # model_name = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
        # adapter_name = "onekq-ai/OneSQL-v0.1-Qwen-32B"
        # self.model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", device_map="auto" ), adapter_name).to("cuda")

        self.model = AutoModelForCausalLM.from_pretrained(model_name , torch_dtype = torch.bfloat16,
                                                           attn_implementation="flash_attention_2", device_map="auto" )#.to(self.device_list[0])#{"": accelerator.process_index}) ###NO DEVICE
        # print(type(self.model))
        # print(self.model.config.quantization_config if hasattr(self.model.config, "quantization_config") else "No quantization config")
        # print(self.model.model.layers[0].self_attn)

        print('Model successfully loaded')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side="left"
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id # updating model config)
        self.semicolon_token_ids = [
            token_id for token, token_id in self.tokenizer.vocab.items() if ';' in token
        ]
        self.semicolon_token_ids.append(self.tokenizer.eos_token_id)
        
    def get_layer_memory(self , layer):
        # Calculate the memory occupied by the parameters of the layer
        param_size = 0
        for param in layer.parameters():
            param_size += param.numel() * param.element_size()  # numel() gives the number of elements in the tensor
                                                                # element_size() gives the size of each element in bytes

        return param_size

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

    #from threading import Thread
    def pipeline_process(self, batch_size , log_file , num_beam=1 , top_k=1 , max_num_token=400 , start_index=0 , end_index=-1 , thread_number=0):
        # works on self.all_prompts_in_1D as data. devides the data into batch_sizes and performs the operation on a batch. It handels the output of each operation on a batch
        #inputs:
            #data: list of prompts
            #batch_size: integer indicating the size of a batch. WARNING: The batch size should be multiple of number of components!
            #operation_on_a_batch: It is a function that is applied on each batch.
            #output_handling: a function responsible for handling the output of each operation on a batch.
        self.length_penalty = 0.5
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
            for propt_idx in range(len(batch)):
                batch[propt_idx] = batch[propt_idx] + '\nSELECT'
                batch[propt_idx] = f"<|im_start|>system\nYou are a SQL expert. Return code only.<|im_end|>\n<|im_start|>user\n{batch[propt_idx]}<|im_end|>\n<|im_start|>assistant\n"
                
            inputs = self.tokenizer(batch, return_tensors="pt" , padding=True).to(self.model.device) #shape(batch_size , sentence_length)

            inputs_ids = inputs.input_ids
            inputs_len = inputs_ids.shape[-1]

            with open(log_file , 'a' ) as f:
                f.write(f'batch_number: {index}- thread num: {thread_number}\n')
                f.write(f'input_ids: {inputs_ids.size()}\n')

            print(f'batch_number: {index}- thread num: {thread_number}\n')
            start = time.time()
            print(f'Thread Num: {thread_number}')
            for i in self.device_list:
                print(f'Start of batch | max {i}:  , {torch.cuda.max_memory_allocated(i)} ')

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    num_beams=num_beam,
                    num_return_sequences=num_beam,  # Return all beams
                    length_penalty=0.5,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache = True#,
                    #eos_token_id = self.semicolon_token_ids
                )
            
            most_probable_gen_text = self.tokenizer.batch_decode(output.sequences[:, inputs_len:], skip_special_tokens = True)#, clean_up_tokenization_spaces = False)
            for sql , prb in zip(most_probable_gen_text, output.sequences_scores):
                print(f'prb: {prb.item()} generated_sqls: ' , sql)
                print('------------------------')
            inputs_log_prob = output.sequences_scores
            print(inputs_log_prob.size())
            
            list_of_inputs_log_prob.extend(inputs_log_prob)
            list_of_batch_text.extend(batch)
            
            with open(log_file , 'a' ) as f:
                # f.write(f'\nThread_num:{thread_number} finished batch index: {index}\n')
                f.write(f'{time.time()-start} \t gen_text_len:{inputs_len}\n')
                # f.write(f'\nThe chosen gen text:  {most_probable_gen_text}\n')
            print('The chosen gen text: ' , most_probable_gen_text, '\n')
            list_of_outputs.extend( most_probable_gen_text )
        print(f'Thread Num: {thread_number}')
        for i in self.device_list:
            print(f'End of process | max {i}:  , {torch.cuda.max_memory_allocated(i)} ')
        # print('\n')
        return list_of_outputs, list_of_inputs_log_prob, list_of_batch_text
        
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
    # send_notif('The job is started in manga-4:)')
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
    # dataset2 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_5-10_max_tokens_8192_max_new_tokens_256.json'
    # dataset3 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_10-15_max_tokens_8192_max_new_tokens_256.json'
    # dataset4 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_15-20_max_tokens_8192_max_new_tokens_256.json'
    # dataset5 = 'codes-1b_BIRD_table_num_5_column_num_6_5-shot_20-25_max_tokens_8192_max_new_tokens_256.json'
    
    # dataset1 = 'SPIDER-TEST_SQL_3-SHOT_0-3_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset2 = 'SPIDER-TEST_SQL_3-SHOT_3-6_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset3 = 'SPIDER-TEST_SQL_3-SHOT_6-9_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset4 = 'SPIDER-TEST_SQL_3-SHOT_9-12_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    # dataset5 = 'SPIDER-TEST_SQL_3-SHOT_12-15_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-2048.json'
    dataset_list = [dataset1 ]#, dataset2 , dataset3 , dataset4 , dataset5]
    batch_size = opt.batch_size * len(dataset_list)

    word_ensemble = LLM_Word_Level_Ensemble(opt.model_name)#, opt.device , opt.other_device)
    word_ensemble.load_prompts_from_datasets(directory , dataset_list , 
                                             starting_index= opt.starting_index, 
                                             ending_index= opt.ending_index)
    # list_of_outputs = word_ensemble.pipeline_process( batch_size , num_beam= opt.num_beam, 
    #                                                  top_k = opt.num_beam )
    list_of_outputs, list_of_inputs_log_prob, list_of_batch_text = word_ensemble.start_pipelines( batch_size , opt.log_file, num_beam=opt.num_beam , top_k=opt.num_beam , num_threads = opt.thread_num)
    with open(opt.output_file , 'wb') as f:
        pkl.dump(list_of_outputs , f)
    with open(opt.output_file[:-4] + '_input_ids.pkl' , 'wb') as f:
        pkl.dump(list_of_inputs_log_prob , f)
    with open(opt.output_file[:-4] + '_batch_text.pkl' , 'wb') as f:
        pkl.dump(list_of_batch_text , f)
    # send_notif(f'The task is done! output is being stored in manga-4 in{opt.output_file}\nProcess time:{time.time()-start_time}s')

